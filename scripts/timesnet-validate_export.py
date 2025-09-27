#!/usr/bin/env python3
"""
Validate exported embeddings (NPZ + meta JSON) and show a minimal
Env-side loading example for PPO consumption.

Usage:
  python scripts/timesnet-validate_export.py /path/to/embeddings_dir

If no path is given, tries to find 'embeddings_meta.json' under ./ or ./notebooks/.
"""
import sys
import json
import numpy as np
from pathlib import Path


def find_meta(base: Path) -> Path:
    candidates = [
        base / 'embeddings_meta.json',
        base / 'notebooks' / 'embeddings_meta.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search
    for p in base.rglob('embeddings_meta.json'):
        return p
    raise FileNotFoundError('embeddings_meta.json not found')


def check_split(npz_path: Path):
    arr = np.load(npz_path, allow_pickle=False)
    required = ['obs', 'X', 'start_idx', 'end_idx']
    for k in required:
        if k not in arr:
            raise AssertionError(f"Missing key '{k}' in {npz_path}")
    obs = arr['obs']
    X = arr['X']
    start_idx = arr['start_idx']
    end_idx = arr['end_idx']
    close = arr['close'] if 'close' in arr.files else None

    # Dtypes
    assert obs.dtype == np.float32, f"obs dtype must be float32, got {obs.dtype}"
    assert X.dtype == np.float32, f"X dtype must be float32, got {X.dtype}"
    assert start_idx.dtype.kind in ('i','u'), f"start_idx must be integer, got {start_idx.dtype}"
    assert end_idx.dtype.kind in ('i','u'), f"end_idx must be integer, got {end_idx.dtype}"
    if close is not None:
        assert close.dtype in (np.float64, np.float32), f"close must be float, got {close.dtype}"

    # Shapes
    T, D = obs.shape
    assert X.shape == (T, D), f"X shape {X.shape} must match obs {obs.shape}"
    assert start_idx.shape == (T,), f"start_idx shape mismatch: {start_idx.shape}"
    assert end_idx.shape == (T,), f"end_idx shape mismatch: {end_idx.shape}"
    if close is not None:
        # At least one valid next step
        valid = end_idx + 1 < len(close)
        assert valid.any(), "No valid next-step index (end_idx+1) within close length"

    print(f"OK {npz_path.name}: obs={obs.shape} start/end len={len(start_idx)} close={None if close is None else len(close)}")
    return T, D


def minimal_env_loader(npz_path: Path):
    """Return (obs, end_idx, close) ready for a simple Env stub."""
    d = np.load(npz_path, allow_pickle=False)
    obs = d['obs'].astype(np.float32, copy=False)
    end_idx = d['end_idx'].astype(np.int64, copy=False)
    close = d['close'].astype(np.float64, copy=False) if 'close' in d.files else None
    return obs, end_idx, close


def demo_reward(obs, end_idx, close):
    """Compute a simple next-bar log return reward for a flat->long action (illustrative)."""
    if close is None:
        print("No 'close' in NPZ; skipping reward demo.")
        return
    t0 = 0
    t1 = min(len(end_idx)-1, 10)
    i0 = end_idx[t0]
    i1 = end_idx[t1]
    r0 = np.log(close[i0+1]/close[i0]) if i0+1 < len(close) else 0.0
    r1 = np.log(close[i1+1]/close[i1]) if i1+1 < len(close) else 0.0
    print(f"Reward demo (next-bar log return): t={t0}->{t0+1} r={r0:.6f}; t={t1}->{t1+1} r={r1:.6f}")


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    meta_path = find_meta(base)
    print(f"Using meta: {meta_path}")
    meta = json.loads(meta_path.read_text())
    splits = meta.get('splits') or meta.get('files')
    if not splits:
        raise AssertionError("Meta missing 'splits' mapping")

    dims = {}
    for name, p in splits.items():
        T, D = check_split(Path(p))
        dims[name] = (T, D)

    # Minimal loader + reward demo on 'train' if available
    if 'train' in splits:
        obs, end_idx, close = minimal_env_loader(Path(splits['train']))
        print(f"Loaded for env: obs={obs.shape}, end_idx={end_idx.shape}, close={None if close is None else close.shape}")
        demo_reward(obs, end_idx, close)

    print("Validation completed.")


if __name__ == '__main__':
    main()

