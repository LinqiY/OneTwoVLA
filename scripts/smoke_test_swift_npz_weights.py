#!/usr/bin/env python3
"""OneTwoVLA smoke test: converted PaliGemma ``.npz`` matches the train checkpoint path.

Runs ``train.init_train_state`` (same as ``train.py``): load ``.npz`` via ``PaliGemmaWeightLoader``,
``check_pytree_equality`` on shapes/dtypes, then jitted merge into the model. No dataset required.

Example (use repo **.venv** for JAX/Flax/etc., and **PYTHONPATH=src**)::

  cd /path/to/OneTwoVLA
  PATH=$(pwd)/.venv/bin:$PATH PYTHONPATH=src python3 scripts/smoke_test_swift_npz_weights.py \\
    --npz /path/to/paligemma-3b-pt-224.npz \\
    --config pifast_w_vlabench_pretrain_primitive_test
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import jax

    import openpi.training.config as _config
    import openpi.training.sharding as sharding
    import openpi.training.weight_loaders as _weight_loaders

    try:
        from scripts.train import init_train_state
    except ImportError:
        from train import init_train_state
except ModuleNotFoundError as e:
    venv_py = _REPO_ROOT / ".venv" / "bin" / "python3"
    raise SystemExit(
        f"Import failed: {e}\n\n"
        "You are not using the OneTwoVLA Python environment (needs flax, jax, openpi deps).\n"
        "Run from the repo root with venv on PATH and PYTHONPATH=src, for example:\n"
        f"  PATH={_REPO_ROOT}/.venv/bin:$PATH PYTHONPATH={_SRC} \\\n"
        f"    {venv_py if venv_py.is_file() else 'python3'} scripts/smoke_test_swift_npz_weights.py --npz ...\n"
        "Or:  cd repo && uv run python scripts/smoke_test_swift_npz_weights.py --npz ..."
    ) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=str, required=True, help="Output of swift_pt_paligemma_to_jax_npz.py.")
    parser.add_argument(
        "--config",
        type=str,
        default="pifast_w_vlabench_pretrain_primitive_test",
        help="Registered TrainConfig / CotrainConfig name (model layout must match the .npz).",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        raise SystemExit(f"npz not found: {npz_path}")

    base = _config.get_config(args.config)
    config = dataclasses.replace(
        base,
        weight_loader=_weight_loaders.PaliGemmaWeightLoader(str(npz_path)),
        exp_name="smoke-swift-npz",
        wandb_enabled=False,
        num_workers=0,
        overwrite=True,
    )

    mesh = sharding.make_mesh(config.fsdp_devices)
    print(f"[smoke] devices={jax.device_count()} config={args.config!r} npz={npz_path}", flush=True)

    train_state, _ = init_train_state(config, jax.random.key(0), mesh, resume=False)
    jax.block_until_ready(train_state)
    print("[smoke] OK: weights loaded and init_train_state succeeded.", flush=True)


if __name__ == "__main__":
    main()
