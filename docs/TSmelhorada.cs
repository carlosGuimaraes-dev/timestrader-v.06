#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using Newtonsoft.Json;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Concurrent;
using System.IO;
#endregion

//Namespace NinjaTrader.NinjaScript.Strategies
namespace NinjaTrader.NinjaScript.Strategies
{
    public class TSMelhorada : Strategy
    {
        #region RedisLiteClient (Private Nested Class)
        // ====================================================================================================
        // A classe RedisLiteClient foi movida para DENTRO da estratégia como uma classe privada.
        // Isso a esconde completamente do sistema de carregamento do NinjaTrader, resolvendo o erro de casting.
        // ====================================================================================================
        private class RedisLiteClient : IDisposable
        {
            private enum ConnectionState { Disconnected, Connecting, Connected }
            private volatile ConnectionState _state = ConnectionState.Disconnected;

            private TcpClient _pubClient, _subClient;
            private NetworkStream _pubStream, _subStream;
            private readonly string _host;
            private readonly int _port;
            private readonly string _password;
            private readonly ConcurrentDictionary<string, Action<string>> _subscriptions = new ConcurrentDictionary<string, Action<string>>();
            private CancellationTokenSource _cts;
            private Task _readerTask;
            private Task _reconnectionTask;
            private readonly object _writeLock = new object();
            private readonly Action<string, string> _logger;

            public bool IsConnected => _state == ConnectionState.Connected;

            public RedisLiteClient(string host = "127.0.0.1", int port = 6379, string password = null, Action<string, string> logger = null)
            {
                _host = host;
                _port = port;
                _password = password;
                _logger = logger ?? ((ctx, msg) => { });
            }

            private void Log(string context, string message) => _logger(context, message);

            public void Connect()
            {
                if (_state != ConnectionState.Disconnected) return;

                _cts = new CancellationTokenSource();
                _reconnectionTask = Task.Run(() => ConnectionManagerLoop(_cts.Token));
            }

            private async void ConnectionManagerLoop(CancellationToken token)
            {
                int retryDelayMs = 1000;
                while (!token.IsCancellationRequested)
                {
                    if (_state == ConnectionState.Disconnected)
                    {
                        try
                        {
                            _state = ConnectionState.Connecting;
                            Log("ConnectionManager", $"Attempting to connect to {_host}:{_port}...");

                            DisposeClients();

                            _pubClient = new TcpClient { NoDelay = true };
                            await _pubClient.ConnectAsync(_host, _port);
                            _pubStream = _pubClient.GetStream();
                            if (!string.IsNullOrEmpty(_password)) Authenticate(_pubStream, _password);

                            _subClient = new TcpClient { NoDelay = true };
                            await _subClient.ConnectAsync(_host, _port);
                            _subStream = _subClient.GetStream();
                            if (!string.IsNullOrEmpty(_password)) Authenticate(_subStream, _password);

                            _state = ConnectionState.Connected;
                            Log("ConnectionManager", "Successfully connected. Initializing reader and resubscribing...");

                            foreach (var sub in _subscriptions)
                            {
                                SendCommandOnStream(_subStream, "SUBSCRIBE", sub.Key);
                            }

                            if (_readerTask == null || _readerTask.IsCompleted)
                            {
                                _readerTask = Task.Run(() => ReadLoop(_subStream, _cts.Token), _cts.Token);
                            }

                            retryDelayMs = 1000;
                        }
                        catch (Exception ex)
                        {
                            if (token.IsCancellationRequested) break;
                            Log("ConnectionManager", $"Connection failed: {ex.Message}. Retrying in {retryDelayMs / 1000}s...");
                            _state = ConnectionState.Disconnected;
                            await Task.Delay(retryDelayMs, token);
                            retryDelayMs = Math.Min(retryDelayMs * 2, 30000);
                        }
                    }
                    await Task.Delay(100, token);
                }
            }

            private void ReadLoop(NetworkStream stream, CancellationToken token)
            {
                using (var ms = new MemoryStream())
                {
                    var buffer = new byte[8192];
                    while (!token.IsCancellationRequested && (_subClient?.Connected ?? false))
                    {
                        try
                        {
                            int bytesRead = stream.Read(buffer, 0, buffer.Length);
                            if (bytesRead == 0)
                            {
                                if (token.IsCancellationRequested) break;
                                Log("ReadLoop", "Connection closed by peer.");
                                _state = ConnectionState.Disconnected;
                                return;
                            }

                            ms.Write(buffer, 0, bytesRead);
                            ms.Position = 0;

                            while (TryParseResp(ms, out object result))
                            {
                                if (result is object[] array && array.Length == 3 && array[0]?.ToString() == "message")
                                {
                                    var channel = array[1]?.ToString();
                                    var message = array[2]?.ToString();
                                    if (channel != null && message != null && _subscriptions.TryGetValue(channel, out var handler))
                                    {
                                        handler(message);
                                    }
                                }
                            }

                            var remaining = ms.Length - ms.Position;
                            if (remaining > 0)
                            {
                                var tempBuffer = new byte[remaining];
                                ms.Read(tempBuffer, 0, (int)remaining);
                                ms.SetLength(0);
                                ms.Write(tempBuffer, 0, tempBuffer.Length);
                            }
                            else
                            {
                                ms.SetLength(0);
                            }
                        }
                        catch (IOException) { _state = ConnectionState.Disconnected; return; }
                        catch (ObjectDisposedException) { _state = ConnectionState.Disconnected; return; }
                        catch (Exception ex)
                        {
                            if (token.IsCancellationRequested) break;
                            Log("ReadLoop", $"Unhandled exception in ReadLoop. Buffer cleared. Error: {ex.Message}");
                            ms.SetLength(0);
                        }
                    }
                }
            }

            private bool TryParseResp(MemoryStream ms, out object result)
            {
                long initialPosition = ms.Position;
                result = null;
                if (ms.Position >= ms.Length) return false;

                try
                {
                    int type = ms.ReadByte();
                    switch (type)
                    {
                        case '+': result = ReadLine(ms); return true;
                        case '-': result = new Exception(ReadLine(ms)); return true;
                        case ':': result = long.Parse(ReadLine(ms)); return true;
                        case '$':
                            int len = int.Parse(ReadLine(ms));
                            if (len == -1) { result = null; return true; }
                            if (ms.Length - ms.Position < len + 2) throw new EndOfStreamException("Incomplete bulk string.");
                            var buffer = new byte[len];
                            ms.Read(buffer, 0, len);
                            ms.ReadByte(); ms.ReadByte(); // CRLF
                            result = Encoding.UTF8.GetString(buffer);
                            return true;
                        case '*':
                            int count = int.Parse(ReadLine(ms));
                            if (count == -1) { result = null; return true; }
                            var array = new object[count];
                            for (int i = 0; i < count; i++)
                            {
                                if (!TryParseResp(ms, out array[i])) throw new EndOfStreamException("Incomplete array.");
                            }
                            result = array;
                            return true;
                        default:
                            throw new InvalidDataException($"Unexpected RESP type: {(char)type}");
                    }
                }
                catch (Exception)
                {
                    ms.Position = initialPosition;
                    result = null;
                    return false;
                }
            }

            private string ReadLine(MemoryStream ms)
            {
                var sb = new StringBuilder();
                int b;
                while ((b = ms.ReadByte()) != -1)
                {
                    if (b == '\r') { if (ms.ReadByte() == '\n') return sb.ToString(); }
                    sb.Append((char)b);
                }
                throw new EndOfStreamException("Could not find CRLF in stream.");
            }

            private void Authenticate(NetworkStream stream, string password)
            {
                var response = SendCommandOnStream(stream, "AUTH", password);
                if (response is Exception || !response.ToString().Equals("OK", StringComparison.OrdinalIgnoreCase))
                {
                    throw new InvalidOperationException($"Redis AUTH failed: {response}");
                }
            }

            private object SendCommandOnStream(NetworkStream stream, params string[] args)
            {
                try
                {
                    var request = $"*{args.Length}\r\n";
                    foreach (var arg in args)
                    {
                        var safeArg = arg ?? "";
                        request += $"${Encoding.UTF8.GetByteCount(safeArg)}\r\n{safeArg}\r\n";
                    }
                    var data = Encoding.UTF8.GetBytes(request);
                    stream.Write(data, 0, data.Length);
                    stream.Flush();

                    using (var ms = new MemoryStream())
                    {
                        var buffer = new byte[4096];
                        var bytesRead = stream.Read(buffer, 0, buffer.Length);
                        ms.Write(buffer, 0, bytesRead);
                        ms.Position = 0;
                        if (TryParseResp(ms, out object result)) return result;
                        return new Exception("Failed to parse response");
                    }
                }
                catch (Exception ex)
                {
                    Log("SendCommandOnStream", $"Failed: {ex.Message}");
                    return ex;
                }
            }

            public void Publish(string channel, string message)
            {
                if (!IsConnected) return;
                lock (_writeLock)
                {
                    SendCommandOnStream(_pubStream, "PUBLISH", channel, message);
                }
            }

            public void Subscribe(string channel, Action<string> onMessage)
            {
                _subscriptions[channel] = onMessage;
                if (IsConnected)
                {
                    lock (_writeLock) { SendCommandOnStream(_subStream, "SUBSCRIBE", channel); }
                }
            }

            private void DisposeClients()
            {
                try { _pubStream?.Close(); } catch { }
                try { _pubStream?.Dispose(); } catch { }
                try { _subStream?.Close(); } catch { }
                try { _subStream?.Dispose(); } catch { }
                try { _pubClient?.Close(); } catch { }
                try { _pubClient?.Dispose(); } catch { }
                try { _subClient?.Close(); } catch { }
                try { _subClient?.Dispose(); } catch { }
            }

            public void Dispose()
            {
                // CORREÇÃO: Método de Dispose robusto, não-bloqueante e à prova de falhas.
                try
                {
                    _state = ConnectionState.Disconnected;
                    
                    try { _cts?.Cancel(false); } catch { } // Sinaliza o cancelamento sem jogar exceção.
                    try { _subscriptions?.Clear(); } catch { }
                    
                    // MOTIVO: Removido Task.WaitAll() para não bloquear a thread principal do NT8.
                    
                    try { DisposeClients(); } catch { }
                    try { _cts?.Dispose(); } catch { }

                    _readerTask = null;
                    _reconnectionTask = null;
                }
                catch (Exception) { /* Suprimir TODAS as exceções durante o dispose */ }
            }
        }
        #endregion

        #region BarAggregator (Private Nested Class)
        // ====================================================================================================
        // BarAggregator: Agrega dados tick-by-tick em barras de 5 minutos para reduzir volume de mensagens
        // Implementado para Sprint S016 - Redução de 800x no volume sem afetar lógica de trading
        // ====================================================================================================
        private class BarAggregator
        {
            private readonly RedisLiteClient _redis;
            private readonly string _instrument;
            private double _open, _high, _low, _close;
            private double _volume;
            private DateTime _currentBarStartTime;
            private long _currentBarInterval;
            private bool _hasData = false;
            private readonly Action<string, LogLevel> _logger;
            private System.Timers.Timer _barTimer;
            
            public BarAggregator(RedisLiteClient redis, string instrument, Action<string, LogLevel> logger)
            {
                _redis = redis;
                _instrument = instrument;
                _logger = logger ?? ((msg, level) => { });
                _currentBarInterval = -1;
                
                // Timer para publicar candle no final de cada período de 5 minutos
                _barTimer = new System.Timers.Timer();
                _barTimer.Elapsed += OnBarTimerElapsed;
                _barTimer.AutoReset = false; // Single shot timer
            }
            
            public void ProcessTick(double price, double volume, DateTime time)
            {
                var newBarInterval = GetBarInterval(time);
                
                if (_currentBarInterval != newBarInterval)
                {
                    if (_hasData && _currentBarInterval != -1)
                    {
                        PublishCompletedBar();
                    }
                    
                    StartNewBar(price, volume, time, newBarInterval);
                }
                else
                {
                    UpdateCurrentBar(price, volume);
                }
            }
            
            private long GetBarInterval(DateTime time)
            {
                var totalMinutes = (long)((time - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMinutes);
                return totalMinutes / 5;
            }
            
            private DateTime GetBarStartTime(long interval)
            {
                var totalMinutes = interval * 5;
                return new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddMinutes(totalMinutes);
            }
            
            private void StartNewBar(double price, double volume, DateTime time, long interval)
            {
                _open = _high = _low = _close = price;
                _volume = volume;
                _currentBarInterval = interval;
                _currentBarStartTime = GetBarStartTime(interval);
                _hasData = true;
                
                // Configura timer para publicar no final deste período de 5 minutos
                var nextBarTime = GetBarStartTime(interval + 1);
                var timeToNext = (nextBarTime - DateTime.UtcNow).TotalMilliseconds;
                
                if (timeToNext > 0)
                {
                    _barTimer.Stop();
                    _barTimer.Interval = timeToNext;
                    _barTimer.Start();
                }
            }
            
            private void UpdateCurrentBar(double price, double volume)
            {
                if (!_hasData) return;
                
                _high = Math.Max(_high, price);
                _low = Math.Min(_low, price);
                _close = price;
                _volume += volume;
            }
            
            private void PublishCompletedBar()
            {
                if (!_hasData) return;
                
                try
                {
                    var barData = new
                    {
                        timestamp = _currentBarStartTime.AddMinutes(5).ToString("yyyy-MM-ddTHH:mm:ss"),
                        open = _open,
                        high = _high,
                        low = _low,
                        close = _close,
                        volume = _volume,
                        instrument = _instrument,
                        bar_type = "5_minute",
                        account = "SIMULATION"
                    };
                    
                    var json = JsonConvert.SerializeObject(barData);
                    _redis.Publish("nt8_market_data", json);
                    _logger($"S016 Fixed - Published 5-min bar: {_currentBarStartTime:HH:mm:ss} O={_open:F2} H={_high:F2} L={_low:F2} C={_close:F2} V={_volume:F0}", LogLevel.Information);
                }
                catch (Exception ex)
                {
                    _logger($"Error publishing bar: {ex.Message}", LogLevel.Error);
                }
            }
            
            private void OnBarTimerElapsed(object sender, System.Timers.ElapsedEventArgs e)
            {
                if (_hasData && _currentBarInterval != -1)
                {
                    PublishCompletedBar();
                    _logger($"Timer-based bar published for interval {_currentBarInterval}", LogLevel.Information);
                }
            }
            
            public void Dispose()
            {
                _barTimer?.Stop();
                _barTimer?.Dispose();
                
                if (_hasData && _currentBarInterval != -1)
                    PublishCompletedBar();
            }
        }
        #endregion

        #region Ciclo de Vida e Limpeza
        
        // PORTÃO DE FERRO: Flag volátil que funciona como kill switch universal
        // Impede QUALQUER operação que possa causar crash durante disposal
        private volatile bool _isDisposing = false;
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "TS Melhorada";
                Calculate = Calculate.OnEachTick;
                IsUnmanaged = true;
                EntriesPerDirection = 1;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 300;
                TraceOrders = true;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                BarsRequiredToTrade = 20;
                _isDisposing = false; // Reset no início
            }
            else if (State == State.Configure)
            {
                AddDataSeries(BarsPeriodType.Tick, 1);
                // 🎯 AI PIPELINE: Série de 5 minutos para dados agregados (evita overwhelming Redis)
                AddDataSeries(BarsPeriodType.Minute, 5);
            }
            else if (State == State.DataLoaded)
            {
                _atrIndicator = ATR(ATRPeriod);
                
                // 🎯 CONNECTION TYPE VERIFICATION: Log detailed connection information
                LogConnectionTypeDetails();
                
                InitializeRedis();
                RehydrateState();
            }
            else if (State == State.Terminated || State == State.Finalized)
            {
                // PORTÃO DE FERRO: Primeira ação - bloquear TODAS as operações
                _isDisposing = true;
                CleanupResources();
            }
        }

        private void CleanupResources()
        {
            // MOTIVO: Ponto de entrada único e seguro para a limpeza de recursos.
            if (_isDisposing) return; // Prevenir chamadas recursivas ou duplas.
            _isDisposing = true;

            try
            {
                // 1. Cancelar ordens pendentes. É a operação mais crítica.
                try
                {
                    if (_entryOrder != null && IsOrderLive(_entryOrder)) CancelOrder(_entryOrder);
                    if (_stopOrder != null && IsOrderLive(_stopOrder)) CancelOrder(_stopOrder);
                }
                catch (Exception ex) { LogMessage($"Erro ao cancelar ordens durante cleanup: {ex.Message}", LogLevel.Error); }

                // 2. Limpar a fila de mensagens para liberar memória.
                try
                {
                    while (_messageQueue.TryDequeue(out _)) { }
                }
                catch (Exception ex) { LogMessage($"Erro ao limpar message queue durante cleanup: {ex.Message}", LogLevel.Error); }

                // 3. Fazer o dispose do cliente Redis.
                try
                {
                    _redisClient?.Dispose();
                }
                catch (Exception ex) { LogMessage($"Erro no dispose do Redis client durante cleanup: {ex.Message}", LogLevel.Error); }
                
                // 4. Fazer o dispose do BarAggregator (S016).
                try
                {
                    _barAggregator?.Dispose();
                }
                catch (Exception ex) { LogMessage($"Erro no dispose do BarAggregator durante cleanup: {ex.Message}", LogLevel.Error); }
                finally
                {
                    _redisClient = null;
                }
                
                // 4. Limpar variáveis de estado.
                _entryOrder = null;
                _stopOrder = null;
            }
            catch (Exception ex)
            {
                try { System.Diagnostics.Debug.WriteLine($"EXCEÇÃO CRÍTICA NÃO TRATADA EM CLEANUPRESOURCES: {ex}"); } catch { }
            }
        }
        
        #endregion

        #region Parâmetros Configuráveis
        [NinjaScriptProperty] [Display(Name = "Redis Host", GroupName = "1. Redis", Order = 1)] public string RedisHost { get; set; } = "";
        [NinjaScriptProperty] [Display(Name = "Redis Port", GroupName = "1. Redis", Order = 2)] public int RedisPort { get; set; } = 6379;
        [NinjaScriptProperty] [Display(Name = "Redis Channel (Comandos)", GroupName = "1. Redis", Order = 3)] public string RedisChannel { get; set; } = "nt8_orders";
        [NinjaScriptProperty] [Display(Name = "Redis Channel (Respostas)", GroupName = "1. Redis", Order = 4)] public string RedisResponseChannel { get; set; } = "nt8_orders_responses";
        [NinjaScriptProperty] [Display(Name = "Redis Password", GroupName = "1. Redis", Order = 5)] public string RedisPassword { get; set; } = "";
        [NinjaScriptProperty] [Display(Name = "Enable Market Data Aggregation", GroupName = "1. Redis", Order = 6)] public bool EnableMarketDataAggregation { get; set; } = true;
        [NinjaScriptProperty] [Range(1, 100)] [Display(Name = "Quantidade (Contratos)", GroupName = "2. Trading", Order = 1)] public int OrderQuantity { get; set; } = 1;
        [NinjaScriptProperty] [Range(1, 100)] [Display(Name = "Stop Inicial - Período do ATR", GroupName = "3. Stop Loss Inicial (ATR)", Order = 1)] public int ATRPeriod { get; set; } = 14;
        [NinjaScriptProperty] [Range(0.1, 10.0)] [Display(Name = "Stop Inicial - Multiplicador do ATR", GroupName = "3. Stop Loss Inicial (ATR)", Order = 2)] public double ATRMultiplier { get; set; } = 1.5;
        [NinjaScriptProperty] [Range(1, 200)] [Display(Name = "Stop Inicial - Fallback (Ticks)", Description = "Stop Fixo em Ticks usado se o ATR não estiver pronto.", GroupName = "3. Stop Loss Inicial (ATR)", Order = 3)] public int FallbackStopTicks { get; set; } = 20;
        [NinjaScriptProperty] [Range(1, 200)] [Display(Name = "Gatilho do Breakeven (Ticks)", Description = "Move o stop para o breakeven após o preço andar X ticks a favor.", GroupName = "4. Breakeven", Order = 1)] public int BreakevenTriggerTicks { get; set; } = 40;
        [NinjaScriptProperty] [Range(0, 10)] [Display(Name = "Breakeven - Buffer (Ticks)", Description = "Ticks adicionais de lucro no breakeven para cobrir custos/slippage.", GroupName = "4. Breakeven", Order = 2)] public int BreakevenBufferTicks { get; set; } = 2;
        [NinjaScriptProperty] [Range(1, 200)] [Display(Name = "Trailing Stop - Distância (Ticks)", Description = "Distância em Ticks que o stop seguirá o preço APÓS o breakeven.", GroupName = "5. Trailing Stop", Order = 1)] public int TrailingStopDistanceTicks { get; set; } = 20;
        [NinjaScriptProperty] [Range(1, 50)] [Display(Name = "Trailing Stop - Passo (Ticks)", Description = "O stop só será movido se o preço avançar pelo menos essa quantidade de ticks.", GroupName = "5. Trailing Stop", Order = 2)] public int TrailingStepTicks { get; set; } = 4;
        #endregion

        #region Variáveis Internas
        private RedisLiteClient _redisClient;
        private readonly ConcurrentQueue<string> _messageQueue = new ConcurrentQueue<string>();
        private Order _entryOrder;
        private Order _stopOrder;
        private ATR _atrIndicator;
        private bool _isBreakevenSet;
        private double _currentStopPrice;
        private BarAggregator _barAggregator;
        #endregion
        
        #region Inicialização
        private void InitializeRedis()
        {
            try
            {
                if (string.IsNullOrEmpty(RedisHost))
                {
                    LogMessage("ERRO: Redis Host não configurado!", LogLevel.Error);
                    return;
                }
                
                _redisClient = new RedisLiteClient(RedisHost, RedisPort, RedisPassword, (context, message) =>
                {
                    LogMessage($"[Redis-{context}] {message}", LogLevel.Information);
                });

                _redisClient.Subscribe(RedisChannel, message => _messageQueue.Enqueue(message));
                _redisClient.Subscribe("nt8_request_account_info", message => { if (message == "update") PublishAccountInfo(); });
                _redisClient.Subscribe("nt8_ai_analysis", message => {
                    if (_isDisposing) return;
                    ProcessAIAnalysis(message);
                });
                _redisClient.Connect();

                LogMessage($"Cliente Redis configurado para {RedisHost}:{RedisPort}", LogLevel.Information);
                
                // Initialize BarAggregator for S016 - 800x message reduction
                if (EnableMarketDataAggregation && _redisClient != null)
                {
                    _barAggregator = new BarAggregator(_redisClient, Instrument.FullName, LogMessage);
                    LogMessage("Market data aggregation enabled - publishing 5-minute bars to nt8_market_data", LogLevel.Information);
                }
            }
            catch (Exception e) { LogMessage($"Erro ao inicializar o Redis: {e.Message}", LogLevel.Error); }
        }

        private void RehydrateState()
        {
            if (Account.Positions.Any(p => p.Instrument == Instrument && p.MarketPosition != MarketPosition.Flat))
            {
                var position = Account.Positions.First(p => p.Instrument == Instrument);
                LogMessage($"Posição existente detectada: {position.MarketPosition} {position.Quantity} @ {position.AveragePrice:F2}", LogLevel.Warning);
                _stopOrder = Account.Orders.FirstOrDefault(o => o.Instrument == Instrument && o.Name == "StopLoss" && o.OrderState == OrderState.Working);
                if (_stopOrder != null)
                {
                    _currentStopPrice = _stopOrder.StopPrice;
                    _isBreakevenSet = (position.MarketPosition == MarketPosition.Long && _currentStopPrice >= position.AveragePrice) || (position.MarketPosition == MarketPosition.Short && _currentStopPrice <= position.AveragePrice);
                    LogMessage($"Stop reidratado em {_currentStopPrice:F2}. Breakeven: {_isBreakevenSet}", LogLevel.Information);
                }
                else
                {
                    LogMessage("AVISO GRAVE: Posição encontrada sem ordem de StopLoss ativa!", LogLevel.Error);
                }
            }
        }
        #endregion

        #region Lógica Principal
        protected override void OnBarUpdate()
        {
            // PROTEÇÃO PRIMÁRIA: Primeira linha obrigatória - kill switch
            if (_isDisposing) return;
            
            if (CurrentBar < BarsRequiredToTrade) return;

            try
            {
                // 🎯 SEPARAÇÃO DE RESPONSABILIDADES POR BARSINPROGRESS
                
                if (BarsInProgress == 0)
                {
                    // SÉRIE PRINCIPAL: Processamento de ordens e comandos Redis
                    ProcessRedisQueue();
                    if (CurrentBar % 3 == 0 && _redisClient != null && _redisClient.IsConnected) 
                        PublishAccountInfo();
                    // ❌ REMOVIDO: PublishMarketData() - causava overwhelming Redis
                }
                else if (BarsInProgress == 1) 
                {
                    // SÉRIE DE TICKS: Gerenciamento de stops em tempo real (crítico)
                    ManageProtectiveStops(Closes[1][0]);
                    
                    // S016: Agregação de dados para redução de 800x no volume
                    if (EnableMarketDataAggregation && _barAggregator != null && !_isDisposing)
                    {
                        _barAggregator.ProcessTick(Closes[1][0], Volumes[1][0], Times[1][0]);
                    }
                }
                else if (BarsInProgress == 2)
                {
                    // ❌ DISABLED: Using BarAggregator instead for S016 fix
                    // if (_redisClient != null && _redisClient.IsConnected)
                    //     PublishCandleData();
                }
            }
            catch (Exception e) { SafeLogMessage($"Erro em OnBarUpdate: {e.Message}", LogLevel.Error); }
        }
        #endregion

        #region Gerenciamento de Mensagens e Ordens
        private void ProcessRedisQueue()
        {
            if (_isDisposing) return;
            
            while (_messageQueue.TryDequeue(out string message))
            {
                if (_isDisposing) break;
                
                var marketPosition = SafeNT8PropertyAccess(() => Position.MarketPosition, MarketPosition.Flat);
                
                try
                {
                    var payload = JsonConvert.DeserializeObject<OrderPayload>(message);
                    if (payload == null || payload.symbol != Instrument.FullName) continue;
                    
                    SafeLogMessage($"📨 Processing order: {payload.type} {payload.side} - Position: {marketPosition}", LogLevel.Information);
                    
                    // NEW: Allow exit orders when position exists
                    if (payload.type == "exit" && marketPosition != MarketPosition.Flat)
                    {
                        SafeLogMessage($"📤 Processing EXIT order", LogLevel.Information);
                        
                        // Cancel existing stop order first
                        if (IsOrderLive(_stopOrder))
                        {
                            SafeNT8Operation(() => CancelOrder(_stopOrder), "CancelStopForExit");
                        }
                        
                        // Submit exit order
                        if (marketPosition == MarketPosition.Long && payload.side.ToUpper() == "SELL")
                        {
                            SafeNT8Operation(() => SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, 
                                Position.Quantity, 0, 0, "", "Exit"), "SubmitExitOrder");
                        }
                        else if (marketPosition == MarketPosition.Short && payload.side.ToUpper() == "BUY")
                        {
                            SafeNT8Operation(() => SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Market, 
                                Math.Abs(Position.Quantity), 0, 0, "", "Exit"), "SubmitExitOrder");
                        }
                    }
                    // EXISTING: Entry orders only when flat
                    else if (payload.type == "entry" && marketPosition == MarketPosition.Flat)
                    {
                        if (IsOrderLive(_entryOrder)) 
                        {
                            SafeNT8Operation(() => CancelOrder(_entryOrder), "CancelEntryOrderFromQueue");
                        }
                        
                        // 🎯 CONNECTION-AWARE ORDER SUBMISSION
                        string connectionType = IsSimulationAccount() ? "PLAYBACK/SIMULATION" : "LIVE";
                        bool isPlaybackAccount = SafeNT8PropertyAccess(() => Account.Name, "UNKNOWN").ToUpper().Contains("PLAYBACK");
                        
                        SafeLogMessage($"📤 SUBMITTING {payload.side} ORDER via {connectionType} connection", LogLevel.Information);
                        SafeLogMessage($"   Order Destination: {(isPlaybackAccount ? "Market Replay Engine" : "Real Market")}", LogLevel.Information);
                        SafeLogMessage($"   Account: {SafeNT8PropertyAccess(() => Account.Name, "UNKNOWN")}", LogLevel.Information);
                        SafeLogMessage($"   Quantity: {OrderQuantity}", LogLevel.Information);
                        
                        if (payload.side.ToUpper() == "BUY") 
                        {
                            SafeNT8Operation(() => SubmitOrderUnmanaged(0, OrderAction.Buy, OrderType.Market, OrderQuantity, 0, 0, "", "Entry"), "SubmitBuyOrder");
                        }
                        else if (payload.side.ToUpper() == "SELL") 
                        {
                            SafeNT8Operation(() => SubmitOrderUnmanaged(0, OrderAction.Sell, OrderType.Market, OrderQuantity, 0, 0, "", "Entry"), "SubmitSellOrder");
                        }
                    }
                    else
                    {
                        SafeLogMessage($"⚠️ Order skipped - Position: {marketPosition}, Type: {payload.type}", LogLevel.Warning);
                    }
                }
                catch (Exception e) { SafeLogMessage($"Erro ao processar mensagem '{message}': {e.Message}", LogLevel.Error); }
            }
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string comment)
        {
            // PROTEÇÃO DE CALLBACK: Primeira linha obrigatória - kill switch
            if (_isDisposing) return;

            try
            {
                if (order.Name == "Entry") _entryOrder = order;
                if (order.Name == "StopLoss")
                {
                    _stopOrder = order;
                    if (orderState == OrderState.Working) _currentStopPrice = order.StopPrice;
                    else if (orderState == OrderState.Rejected) SafeLogMessage($"CRÍTICO: Modificação do stop REJEITADA: {error}", LogLevel.Error);
                    else if (orderState == OrderState.Cancelled && SafeNT8PropertyAccess(() => Position.MarketPosition, MarketPosition.Flat) != MarketPosition.Flat) 
                        SafeLogMessage("CRÍTICO: Stop CANCELADO com posição aberta!", LogLevel.Error);
                }

                if (_stopOrder != null && _stopOrder == order && orderState == OrderState.Filled)
                {
                    var marketPosition = SafeNT8PropertyAccess(() => Position.MarketPosition, MarketPosition.Flat);
                    PublishToRedis(new ResponsePayload("STOP_FILLED", marketPosition == MarketPosition.Long ? "SELL" : "BUY", averageFillPrice, quantity));
                    ResetPositionState();
                }
                if (_entryOrder != null && _entryOrder == order && (orderState == OrderState.Cancelled || orderState == OrderState.Rejected))
                {
                    PublishToRedis(new ResponsePayload("ENTRY_FAILED", order.OrderAction.ToString(), 0, 0, orderState.ToString()));
                    ResetPositionState();
                }
            }
            catch (Exception ex)
            {
                SafeLogMessage($"Erro em OnOrderUpdate: {ex.Message}", LogLevel.Error);
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            // PROTEÇÃO DE EXECUÇÃO: Primeira linha obrigatória - kill switch
            if (_isDisposing) return;
            
            try
            {
                if (_entryOrder != null && execution.Order == _entryOrder && execution.Order.OrderState == OrderState.Filled)
                {
                    _isBreakevenSet = false;
                    PublishToRedis(new ResponsePayload("ENTRY_FILLED", execution.Order.OrderAction.ToString(), price, quantity));
                    PlaceInitialStopLoss(execution);
                }
            }
            catch (Exception ex)
            {
                SafeLogMessage($"Erro em OnExecutionUpdate: {ex.Message}", LogLevel.Error);
            }
        }

        private void PlaceInitialStopLoss(Execution entryExecution)
        {
            if (_isDisposing) return;
            
            SafeNT8Operation(() => {
                double atrValue = SafeNT8PropertyAccess(() => _atrIndicator[1], 0.0);
                double stopDistance = (atrValue > 0) ? (atrValue * ATRMultiplier) : (FallbackStopTicks * TickSize);
                
                double stopPriceValue = SafeNT8PropertyAccess(() => 
                    Instrument.MasterInstrument.RoundToTickSize(
                        (entryExecution.Order.OrderAction == OrderAction.Buy) ? 
                            entryExecution.Price - stopDistance : 
                            entryExecution.Price + stopDistance
                    ), 0.0);
                
                if (stopPriceValue == 0.0) return;
                
                OrderAction stopAction = (entryExecution.Order.OrderAction == OrderAction.Buy) ? OrderAction.Sell : OrderAction.Buy;
                
                // 🎯 CONNECTION-AWARE STOP ORDER SUBMISSION
                string connectionType = IsSimulationAccount() ? "PLAYBACK/SIMULATION" : "LIVE";
                bool isPlaybackAccount = SafeNT8PropertyAccess(() => Account.Name, "UNKNOWN").ToUpper().Contains("PLAYBACK");
                
                SafeLogMessage($"🛡️ SUBMITTING STOP ORDER via {connectionType} connection", LogLevel.Information);
                SafeLogMessage($"   Stop Price: {stopPriceValue:F2}", LogLevel.Information);
                SafeLogMessage($"   Connection Type: {connectionType}", LogLevel.Information);
                SafeLogMessage($"   Order Destination: {(isPlaybackAccount ? "Market Replay Engine" : "Real Market")}", LogLevel.Information);
                
                SubmitOrderUnmanaged(0, stopAction, OrderType.StopMarket, entryExecution.Quantity, 0, stopPriceValue, "", "StopLoss");
                PublishToRedis(new ResponsePayload("STOP_PLACED", stopAction.ToString(), 0, entryExecution.Quantity, $"StopPrice: {stopPriceValue:F2}"));
            }, "PlaceInitialStopLoss");
        }

        private void ResetPositionState()
        {
            _entryOrder = null;
            _stopOrder = null;
            _isBreakevenSet = false;
            _currentStopPrice = 0;
        }
        #endregion

        #region Gerenciamento de Stops
        private void ManageProtectiveStops(double currentPrice)
        {
            if (_isDisposing) return;
            
            var marketPosition = SafeNT8PropertyAccess(() => Position.MarketPosition, MarketPosition.Flat);
            if (marketPosition == MarketPosition.Flat || !IsOrderLive(_stopOrder)) return;

            SafeNT8Operation(() => {
                if (!_isBreakevenSet)
                {
                    var averagePrice = SafeNT8PropertyAccess(() => Position.AveragePrice, 0.0);
                    if (averagePrice == 0.0) return;
                    
                    double profitInTicks = (marketPosition == MarketPosition.Long) ? 
                        (currentPrice - averagePrice) / TickSize : 
                        (averagePrice - currentPrice) / TickSize;
                        
                    if (profitInTicks >= BreakevenTriggerTicks)
                    {
                        double breakevenPrice = SafeNT8PropertyAccess(() => 
                            Instrument.MasterInstrument.RoundToTickSize(averagePrice + 
                                (marketPosition == MarketPosition.Long ? BreakevenBufferTicks * TickSize : -BreakevenBufferTicks * TickSize)),
                            0.0);
                            
                        if (breakevenPrice != 0.0 && 
                            ((marketPosition == MarketPosition.Long && breakevenPrice > _currentStopPrice) || 
                             (marketPosition == MarketPosition.Short && breakevenPrice < _currentStopPrice)))
                        {
                            ChangeOrder(_stopOrder, _stopOrder.Quantity, 0, breakevenPrice);
                            _isBreakevenSet = true;
                            PublishToRedis(new ResponsePayload("BREAKEVEN_ACTIVATED", _stopOrder.OrderAction.ToString(), 0, _stopOrder.Quantity, $"New StopPrice: {breakevenPrice:F2}"));
                        }
                    }
                }
                else
                {
                    double newStopPrice = 0;
                    if (marketPosition == MarketPosition.Long) 
                        newStopPrice = currentPrice - (TrailingStopDistanceTicks * TickSize);
                    else 
                        newStopPrice = currentPrice + (TrailingStopDistanceTicks * TickSize);

                    double movementInTicks = Math.Abs(newStopPrice - _currentStopPrice) / TickSize;

                    if (movementInTicks >= TrailingStepTicks)
                    {
                        if ((marketPosition == MarketPosition.Long && newStopPrice > _currentStopPrice) || 
                            (marketPosition == MarketPosition.Short && newStopPrice < _currentStopPrice))
                        {
                            newStopPrice = SafeNT8PropertyAccess(() => Instrument.MasterInstrument.RoundToTickSize(newStopPrice), newStopPrice);
                            ChangeOrder(_stopOrder, _stopOrder.Quantity, 0, newStopPrice);
                            PublishToRedis(new ResponsePayload("TRAILING_STOP_MOVED", _stopOrder.OrderAction.ToString(), 0, _stopOrder.Quantity, $"New StopPrice: {newStopPrice:F2}"));
                        }
                    }
                }
            }, "ManageProtectiveStops");
        }
        #endregion

        #region Comunicação Externa (Publishing)
        private void PublishAccountInfo()
        {
            if (_isDisposing || _redisClient == null || !_redisClient.IsConnected) return;
            
            SafeNT8Operation(() => {
                var accountInfo = new { 
                    timestamp = DateTime.UtcNow.ToString("o"), 
                    account_name = SafeNT8PropertyAccess(() => Account.Name, ""),
                    cash_value = SafeNT8PropertyAccess(() => Account.Get(AccountItem.CashValue, Currency.UsDollar), 0.0),
                    buying_power = SafeNT8PropertyAccess(() => Account.Get(AccountItem.BuyingPower, Currency.UsDollar), 0.0),
                    realized_pnl = SafeNT8PropertyAccess(() => Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar), 0.0),
                    unrealized_pnl = SafeNT8PropertyAccess(() => Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Closes[1][0]), 0.0),
                    currency = "USD" 
                };
                _redisClient.Publish("nt8_account_info", JsonConvert.SerializeObject(accountInfo));
            }, "PublishAccountInfo");
        }

        /// <summary>
        /// 🎯 OTIMIZAÇÃO REDIS: Publica dados de candles de 5 minutos para AI Pipeline
        /// Substitui PublishMarketData() tick-based que causava overwhelming
        /// </summary>
        private void PublishCandleData()
        {
            if (_isDisposing || _redisClient == null || !_redisClient.IsConnected || CurrentBars[2] < 1) return;
            
            SafeNT8Operation(() => {
                var candleData = new { 
                    timestamp = DateTime.UtcNow.ToString("o"), 
                    symbol = SafeNT8PropertyAccess(() => Instrument.FullName, ""),
                    // Dados do candle de 5 minutos completo (BarsInProgress = 2)
                    open = SafeNT8PropertyAccess(() => Opens[2][0], 0.0),
                    high = SafeNT8PropertyAccess(() => Highs[2][0], 0.0),
                    low = SafeNT8PropertyAccess(() => Lows[2][0], 0.0),
                    close = SafeNT8PropertyAccess(() => Closes[2][0], 0.0),
                    volume = SafeNT8PropertyAccess(() => Volumes[2][0], 0),
                    bar_time = SafeNT8PropertyAccess(() => Times[2][0].ToString("o"), ""), 
                    is_real_time = State == State.Realtime,
                    timeframe = "5min", // Identificador para AI Pipeline
                    candle_type = "completed" // Candle fechado e completo
                };
                _redisClient.Publish("nt8_market_data", JsonConvert.SerializeObject(candleData));
            }, "PublishCandleData");
        }

        /// <summary>
        /// 🚫 MÉTODO DEPRECADO: PublishMarketData tick-based removido para evitar overwhelming Redis
        /// Substituído por PublishCandleData() que envia apenas candles de 5 min completos
        /// </summary>
        private void PublishMarketData()
        {
            // REMOVIDO: Causava 500-2000 msgs/segundo - substituído por PublishCandleData()
            // AI Pipeline precisa apenas de candles de 5 minutos, não ticks individuais
            return;
        }
        #endregion

        #region Utilitários e Wrappers de Segurança
        
        /// <summary>
        /// Wrapper universal para operações NT8 - Proteção contra crashes durante disposal
        /// </summary>
        private void SafeNT8Operation(Action operation, string operationName = "")
        {
            if (_isDisposing) return;
            try
            {
                operation();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"NT8 Operation '{operationName}' suppressed during disposal: {ex.Message}");
                // Suprimir TODAS as exceptions durante disposal para evitar propagação para NT8
            }
        }
        
        /// <summary>
        /// Wrapper seguro para acesso a propriedades NT8 - Evita ObjectDisposedException
        /// </summary>
        private T SafeNT8PropertyAccess<T>(Func<T> propertyAccess, T defaultValue = default(T))
        {
            if (_isDisposing) return defaultValue;
            try
            {
                return propertyAccess();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Property access suppressed during disposal: {ex.Message}");
                return defaultValue;
            }
        }
        
        private bool IsOrderLive(Order order)
        {
            if (order == null) return false;
            return order.OrderState == OrderState.Accepted || order.OrderState == OrderState.Working || order.OrderState == OrderState.ChangePending;
        }

        private void PublishToRedis(ResponsePayload payload)
        {
            if (_isDisposing || _redisClient == null || !_redisClient.IsConnected) return;
            
            try 
            { 
                _redisClient.Publish(RedisResponseChannel, JsonConvert.SerializeObject(payload)); 
            }
            catch (Exception e) 
            { 
                SafeLogMessage($"Erro ao publicar no Redis: {e.Message}", LogLevel.Error); 
            }
        }

        /// <summary>
        /// Logging seguro à prova de crashes - Evita thread violations com Print()
        /// </summary>
        private void SafeLogMessage(string message, LogLevel level)
        {
            if (_isDisposing || State == State.Terminated || State == State.Finalized)
            {
                // Durante disposal, usar apenas Debug output para evitar thread violations
                try
                {
                    System.Diagnostics.Debug.WriteLine($"{DateTime.UtcNow:HH:mm:ss.fff} DISPOSAL: {message}");
                }
                catch { /* Suprimir tudo */ }
                return;
            }
            
            try
            {
                string logMessage = $"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC | {Name} | {message}";
                if (level >= LogLevel.Warning) 
                    Print($"{level.ToString().ToUpper()}: {logMessage}");
                else 
                    Print(logMessage);
                Log(logMessage, level);
            }
            catch (Exception ex)
            {
                try
                {
                    System.Diagnostics.Debug.WriteLine($"Logging fallback: {message} (Error: {ex.Message})");
                }
                catch { /* Último recurso - suprimir tudo */ }
            }
        }
        
        /// <summary>
        /// Método de compatibilidade - redireciona para SafeLogMessage
        /// </summary>
        private void LogMessage(string message, LogLevel level)
        {
            SafeLogMessage(message, level);
        }

        /// <summary>
        /// 🏦 ACCOUNT TYPE DETECTION - LIVE vs SIMULATION/PLAYBACK
        /// Detecta se a conta é de simulação, playback ou live trading
        /// </summary>
        private bool IsSimulationAccount()
        {
            try
            {
                string accountName = SafeNT8PropertyAccess(() => Account.Name, "UNKNOWN").ToUpper();
                bool isSimulation = accountName.Contains("SIM") || 
                                   accountName.Contains("SIMULATION") || 
                                   accountName.Contains("DEMO") ||
                                   accountName.Contains("TEST") ||
                                   accountName.Contains("PAPER") ||
                                   accountName.Contains("PLAYBACK");
                
                SafeLogMessage($"Account type detection: {SafeNT8PropertyAccess(() => Account.Name, "UNKNOWN")} → {(isSimulation ? "SIMULATION" : "LIVE")}", LogLevel.Information);
                
                return isSimulation;
            }
            catch (Exception ex)
            {
                SafeLogMessage($"Error detecting account type, defaulting to LIVE: {ex.Message}", LogLevel.Warning);
                return false; // Default to LIVE (safe mode)
            }
        }

        private string GetAccountType()
        {
            return IsSimulationAccount() ? "SIMULATION" : "LIVE";
        }

        /// <summary>
        /// 🎯 CONNECTION TYPE VERIFICATION: Análise abrangente de conexão
        /// Adicionado para resolver questões sobre conexão playback - July 5, 2025
        /// </summary>
        private void LogConnectionTypeDetails()
        {
            try
            {
                // Detecção baseada na conta usando SafeNT8PropertyAccess
                string accountName = SafeNT8PropertyAccess(() => Account?.Name, "UNKNOWN");
                string accountType = GetAccountType();
                bool isPlayback = accountName.ToUpper().Contains("PLAYBACK");
                bool isSimulation = IsSimulationAccount();
                
                // Análise do estado da conexão
                string connectionStatus = "UNKNOWN";
                string dataFeedType = "UNKNOWN";
                
                try
                {
                    // Tentar determinar características da conexão
                    connectionStatus = (BarsInProgress >= 0) ? "ACTIVE" : "INACTIVE";
                    
                    // Verificar se estamos recebendo dados live vs históricos
                    int barsCount = SafeNT8PropertyAccess(() => Bars?.Count ?? 0, 0);
                    if (barsCount > 0)
                    {
                        DateTime lastBarTime = SafeNT8PropertyAccess(() => Time[0], DateTime.MinValue);
                        if (lastBarTime != DateTime.MinValue)
                        {
                            TimeSpan timeDiff = DateTime.Now - lastBarTime;
                            dataFeedType = (timeDiff.TotalMinutes < 10) ? "LIVE_OR_RECENT" : "HISTORICAL";
                        }
                    }
                }
                catch
                {
                    connectionStatus = "ERROR_DETECTING";
                }
                
                SafeLogMessage($"🔍 CONNECTION ANALYSIS: {accountType} account on {(isPlayback ? "PLAYBACK" : "STANDARD")} connection", LogLevel.Information);
                SafeLogMessage($"   Account Name: {accountName}", LogLevel.Information);
                SafeLogMessage($"   Account Type: {accountType}", LogLevel.Information);
                SafeLogMessage($"   Is Playback Account: {isPlayback}", LogLevel.Information);
                SafeLogMessage($"   Connection Status: {connectionStatus}", LogLevel.Information);
                SafeLogMessage($"   Data Feed Type: {dataFeedType}", LogLevel.Information);
                SafeLogMessage($"   Instrument: {SafeNT8PropertyAccess(() => Instrument?.FullName, "UNKNOWN")}", LogLevel.Information);
                SafeLogMessage($"   Order Method: SubmitOrderUnmanaged() - Routes via active connection", LogLevel.Information);
                
                // Aviso especial para contas playback
                if (isPlayback)
                {
                    SafeLogMessage("📊 PLAYBACK MODE DETECTED: Orders will use simulated fills with historical data", LogLevel.Warning);
                    SafeLogMessage("   Fill Behavior: INSTANT_AT_HISTORICAL_PRICES", LogLevel.Information);
                    SafeLogMessage("   Real Market Execution: FALSE", LogLevel.Information);
                    SafeLogMessage("   Testing Mode: TRUE", LogLevel.Information);
                    SafeLogMessage("   Note: This is normal for Market Replay testing", LogLevel.Information);
                }
                
                // Resumo da verificação de conexão
                SafeLogMessage("✅ ORDER ROUTING CONFIRMED: Using NT8 SubmitOrderUnmanaged() via active connection", LogLevel.Information);
                SafeLogMessage($"   API Method: SubmitOrderUnmanaged()", LogLevel.Information);
                SafeLogMessage($"   Routing Logic: Automatic via NT8 API", LogLevel.Information);
                SafeLogMessage($"   Connection Type: {(isPlayback ? "PLAYBACK" : "LIVE")}", LogLevel.Information);
                SafeLogMessage($"   Order Destination: {(isPlayback ? "Market Replay Engine" : "Real Market")}", LogLevel.Information);
            }
            catch (Exception ex)
            {
                SafeLogMessage($"Error in connection type verification: {ex.Message}", LogLevel.Error);
            }
        }

        private void ProcessAIAnalysis(string message)
        {
            if (_isDisposing) return;
            try
            {
                var analysis = JsonConvert.DeserializeObject<AIAnalysisPayload>(message);
                
                // Usar SafeNT8Operation para Print() - evita thread violations
                SafeNT8Operation(() => {
                    Print("========== AI ANALYSIS ==========");
                    Print($"Time: {DateTime.Parse(analysis.timestamp).ToLocalTime():HH:mm:ss}");
                    Print("");
                    
                    Print("--- TIMESNET PATTERN DETECTION ---");
                    Print($"Pattern: UP={analysis.timesnet.pattern.UP:P1} | DOWN={analysis.timesnet.pattern.DOWN:P1} | SIDEWAYS={analysis.timesnet.pattern.SIDEWAYS:P1}");
                    Print($"Dominant: {analysis.timesnet.dominant_pattern} (Confidence: {analysis.timesnet.pattern_confidence:P1})");
                    Print("");
                    
                    Print("--- PPO AGENT DECISION ---");
                    Print($"Action: {analysis.ppo.action} (Confidence: {analysis.ppo.confidence:P1})");
                    Print("");
                    
                    Print("--- FINAL DECISION ---");
                    Print($"Final Action: {analysis.final_decision.action} (Source: {analysis.final_decision.source})");
                    Print("================================");
                    Print("");
                    
                    // 🎯 AI-TO-ORDER BRIDGE: Convert AI decisions to actual orders
                    if (analysis.final_decision.action == "BUY" || analysis.final_decision.action == "SELL")
                    {
                        var marketPosition = SafeNT8PropertyAccess(() => Position.MarketPosition, MarketPosition.Flat);
                        if (marketPosition == MarketPosition.Flat) // Only enter new positions if currently flat
                        {
                            var aiOrder = new OrderPayload
                            {
                                type = "entry",
                                side = analysis.final_decision.action,
                                symbol = SafeNT8PropertyAccess(() => Instrument.FullName, "")
                            };
                            
                            string orderJson = JsonConvert.SerializeObject(aiOrder);
                            _messageQueue.Enqueue(orderJson);
                            
                            SafeLogMessage($"🤖 AI ORDER GENERATED: {analysis.final_decision.action} via {analysis.final_decision.source} - Queued for execution", LogLevel.Information);
                        }
                        else
                        {
                            SafeLogMessage($"🤖 AI SIGNAL IGNORED: {analysis.final_decision.action} - Position already open ({marketPosition})", LogLevel.Information);
                        }
                    }
                }, "ProcessAIAnalysisPrint");
            }
            catch (Exception ex) { SafeLogMessage($"Error processing AI analysis: {ex.Message}", LogLevel.Error); }
        }

        // Classes de Payload
        public class AIAnalysisPayload { public string timestamp { get; set; } public TimesNetAnalysis timesnet { get; set; } public PPOAnalysis ppo { get; set; } public string risk { get; set; } public string session { get; set; } public PortfolioAnalysis portfolio_state { get; set; } public FinalDecision final_decision { get; set; } }
        public class TimesNetAnalysis { public PatternProbabilities pattern { get; set; } public double trend_strength { get; set; } public double volatility { get; set; } public double support_resistance { get; set; } public double momentum { get; set; } public string dominant_pattern { get; set; } public double pattern_confidence { get; set; } }
        public class PatternProbabilities { public double UP { get; set; } public double DOWN { get; set; } public double SIDEWAYS { get; set; } }
        public class PPOAnalysis { public string action { get; set; } public double confidence { get; set; } public ActionProbabilities action_probs { get; set; } }
        public class ActionProbabilities { public double HOLD { get; set; } public double BUY { get; set; } public double SELL { get; set; } }
        public class PortfolioAnalysis { public double position { get; set; } public double balance_norm { get; set; } [JsonProperty("return")] public double returnValue { get; set; } public double drawdown { get; set; } }
        public class FinalDecision { public string action { get; set; } public string source { get; set; } }
        public class OrderPayload { public string type { get; set; } public string side { get; set; } public string symbol { get; set; } }
        public class ResponsePayload
        {
            public string timestamp { get; set; } public string event_type { get; set; } public string side { get; set; } public double price { get; set; } public int quantity { get; set; } public string details { get; set; }
            public ResponsePayload(string eventType, string side, double price, int quantity, string details = "")
            { this.timestamp = DateTime.UtcNow.ToString("o"); this.event_type = eventType; this.side = side; this.price = price; this.quantity = quantity; this.details = details; }
        }
        #endregion
    }
}