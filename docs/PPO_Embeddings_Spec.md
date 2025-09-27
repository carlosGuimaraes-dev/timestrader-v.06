**Overview**
- TimesNet acts only as an embedding generator. The PPO agent evaluates embedding quality implicitly by trading performance (PnL/returns), not by offline probes.
- This doc captures the PPO brainstorm (design of observations, actions, reward) and the agreed export format that the PPO Env will consume.

**PPO Requirements (SB3)**
- Env API: `reset() -> obs`, `step(action) -> (obs, reward, done, info)`.
- Spaces: `observation_space = Box(shape=(D,), dtype=float32)`; `action_space` defined by the policy (see below).
- Observations are `np.float32`; rewards are `float`.
- SB3 does not constrain on-disk formats; we standardize our export so the Env can reconstruct observations and rewards without re-opening the raw CSV.

**Env Design (Brainstorm)**
- Observation `obs_t` combines:
  - TimesNet embedding `E_t` (recommended `pooling='avg'`, D≈256).
  - Portfolio state (not stored in export; computed inside Env): e.g., current position in {-1,0,1}, entry_price_rel, unreal_pnl_rel, equity_rel, exposure, steps_in_pos.
  - Price features (computed inside Env from `close`): shift-1 price, previous return, rolling vol.
- Action (target position): `Discrete(3)` representing Short(-1), Flat(0), Long(+1). Live adapter translates into NinjaTrader orders BUY/SELL/HOLD depending on transitions.
- Reward (per-step): `pos_t * log(close[t+1]/close[t]) - costs`, with costs applied when changing position (commission_bps, slippage_bps). Position persists until policy changes it.
- No look-ahead: embedding at `t` maps to `end_idx[t]` in the split; rewards use `close[end_idx[t]+1]` relative to `close[end_idx[t]]`. Final timesteps without a next price are excluded/terminated via time-limit.

**Export Format (Definitive)**
- One directory per experiment, e.g. `.../embeddings/`.
- One NPZ per split (`train`, `val`, `test`) and a `embeddings_meta.json` describing the set.

NPZ content per split (keys and dtypes)
- `obs`: float32 [T, D] — embeddings ready for PPO observation.
- `X`: float32 [T, D] — alias of `obs` for compatibility.
- `start_idx`: int64 [T] — window start index in the split.
- `end_idx`: int64 [T] — window end index in the split; this indexes the `close` vector below.
- `close`: float64 [N] — price series of the split (same scaling as CSV). Used by the Env to compute rewards and price-based features.
- `timestamp` (optional): str [N] ISO8601 or int64 epoch for logging/live alignment.

Meta JSON `embeddings_meta.json`
- Minimal fields:
  - `schema_version`: 1
  - `created_at`: ISO8601 (UTC)
  - `seq_len`: int
  - `window_stride`: int
  - `pooling`: 'avg' or 'flatten'
  - `embed_dim`: int — final observation dimension D (after pooling)
  - `embed_hw`: [H, W] — spatial map size before pooling
  - `features`: list of input feature names used by TimesNet
  - `normalization`: 'StandardScaler' or 'MinMaxScaler'
  - `timesnet_checkpoint_path`: string path
  - `splits`: mapping like `{"train": "/.../train_emb_avg.npz", ...}`
  - `index_semantics`: "end_idx indexes the 'close' vector of the same split"
- Optional fields:
  - `instrument`, `timezone`, `commit`

**Indexing Semantics**
- `end_idx[t]` indexes the `close` vector of the same split. The next-bar reward uses `close[end_idx[t]+1] / close[end_idx[t]]`.
- The Env must mark invalid timesteps where `end_idx[t]+1 >= len(close)` (end-of-episode).

**Why PPO Evaluates Embeddings (not LR/Probes)**
- PPO learns its own non-linear policy on top of the embedding, combining portfolio state and price context. Offline probes (LR/GBDT) are optional diagnostics but are not authoritative; final evaluation is PnL/returns.

**Minimal Env-side Loading Example**
- Use `scripts/timesnet-validate_export.py` to validate shapes/dtypes and to see a simple loader.
- Loader outline:
  - Load `.npz`: `obs` (float32), `end_idx` (int64), `close` (float).
  - Build observation directly from `obs[t]` and augment with portfolio state inside the Env.
  - Compute rewards from `close` using `end_idx` with no look-ahead.

**Suggested PPO Setup (Reference)**
- Policy: `MlpPolicy`, with `observation_space=Box(D,)`, `action_space=Discrete(3)`.
- Common hyperparams: `n_steps=2048`, `batch_size=256`, `gamma=0.99`, `gae_lambda=0.95`, `learning_rate=3e-4`, `clip_range=0.2`.
- Use `Monitor`, `EvalCallback` on the `val` split, and consider `VecNormalize` for observation/reward normalization.
