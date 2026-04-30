# scripts/smoke_test_cotrain.py

import argparse
import dataclasses
import math
import time
from typing import Any

import jax
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

try:
    from train_cotrain import init_train_state, compute_grad_vla, compute_grad_vl, apply_combined_update
except ImportError:
    from scripts.train_cotrain import init_train_state, compute_grad_vla, compute_grad_vl, apply_combined_update


def _shape(x: Any):
    return getattr(x, "shape", None)


def _dtype(x: Any):
    return getattr(x, "dtype", None)


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _make_smoke_config(config: _config.TrainConfig) -> _config.TrainConfig:
    """
    为 smoke test 生成一个更轻量的 config：
    - 关闭 wandb
    - 不用预训练权重（只为了测试能不能跑通）
    - num_workers=0
    - 缩小 batch_size，但保证 VL batch 仍满足 device 整除约束
    """
    num_devices = jax.device_count()
    cotrain_ratio = getattr(config, "cotrain_ratio", 0.25)

    min_batch_for_vl = int(math.ceil(num_devices / cotrain_ratio))
    smoke_batch_size = int(math.ceil(min_batch_for_vl / num_devices) * num_devices)
    smoke_batch_size = max(smoke_batch_size, num_devices)

    return dataclasses.replace(
        config,
        batch_size=smoke_batch_size,
        val_batch_size=smoke_batch_size,
        num_workers=0,
        use_val_dataset=False,
        wandb_enabled=False,
        weight_loader=_weight_loaders.NoOpWeightLoader(),
    )


def smoke_test_vl_dataset(config: _config.TrainConfig):
    from openpi.policies.parquet_vl_dataset import ParquetVQADataset, is_parquet_vl_group
    from openpi.policies.vl_dataset import ShareRobotVQADataset

    vl_data_config = config.vl_data.create(config.assets_dirs, config.model)

    g0 = vl_data_config.vl_groups[0]
    if is_parquet_vl_group(g0):
        ds = ParquetVQADataset(
            g0.parquet_paths,
            g0.parquet_source_id or "",
            image_root=g0.image_root,
        )
    else:
        ds = ShareRobotVQADataset(json_paths=g0.json_paths, image_root=g0.image_root)

    _assert(len(ds) > 0, "VL dataset is empty")

    sample = ds[0]

    _assert("images" in sample, "VL sample must contain 'images'")
    _assert("question" in sample, "VL sample must contain 'question'")
    _assert("answer" in sample, "VL sample must contain 'answer'")

    images = sample["images"]
    question = sample["question"]
    answer = sample["answer"]

    _assert(isinstance(images, list), "'images' must be a list")
    _assert(len(images) > 0, "'images' must be non-empty")
    _assert(isinstance(question, str), "'question' must be str")
    _assert(isinstance(answer, str), "'answer' must be str")

    img0 = images[0]
    _assert(img0.ndim == 3 and img0.shape[-1] == 3, "image[0] must be HWC RGB")
    _assert(str(img0.dtype) == "uint8", "image[0] dtype must be uint8")


def smoke_test_vl_transform(config: _config.TrainConfig):
    from openpi.policies.parquet_vl_dataset import ParquetVQADataset, is_parquet_vl_group
    from openpi.policies.vl_dataset import ShareRobotVQADataset

    vl_data_config = config.vl_data.create(config.assets_dirs, config.model)

    g0 = vl_data_config.vl_groups[0]
    if is_parquet_vl_group(g0):
        ds = ParquetVQADataset(
            g0.parquet_paths,
            g0.parquet_source_id or "",
            image_root=g0.image_root,
        )
    else:
        ds = ShareRobotVQADataset(json_paths=g0.json_paths, image_root=g0.image_root)
    raw_sample = ds[0]

    transform = _transforms.compose(
        [
            *vl_data_config.repack_transforms.inputs,
            *vl_data_config.data_transforms.inputs,
            _transforms.Normalize(vl_data_config.norm_stats, use_quantiles=vl_data_config.use_quantile_norm),
            *vl_data_config.model_transforms.inputs,
        ]
    )

    sample = transform(raw_sample)

    required_keys = [
        "state",
        "image",
        "image_mask",
        "actions",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
    ]
    for k in required_keys:
        _assert(k in sample, f"Missing transformed key: {k}")

    _assert(sample["state"].shape == (config.model.action_dim,), "state shape mismatch")
    _assert(
        sample["actions"].shape == (config.model.action_horizon, config.model.action_dim),
        "dummy actions shape mismatch",
    )
    _assert(
        sample["tokenized_prompt"].shape == (config.model.max_token_len,),
        "tokenized_prompt shape mismatch",
    )


def smoke_test_vla_loader(config: _config.TrainConfig):
    loader, _ = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_workers=0,
    )
    obs, actions = next(iter(loader))

    _assert(actions.ndim == 3, "VLA actions must be rank-3")
    _assert(obs.state.ndim == 2, "obs.state must be batched")
    _assert(obs.tokenized_prompt.ndim == 2, "obs.tokenized_prompt must be batched")


def smoke_test_vl_loader(config: _config.TrainConfig):
    loader = _data_loader.create_vl_data_loader(
        config,
        shuffle=False,
        num_workers=0,
    )
    obs, actions = next(iter(loader))
    print("VL actions dtype/shape/type:", actions.dtype, actions.shape, type(actions), flush=True)

    _assert(actions.ndim == 3, "VL dummy actions must be rank-3")
    _assert(obs.state.ndim == 2, "VL obs.state must be batched")
    _assert(obs.tokenized_prompt.ndim == 2, "VL obs.tokenized_prompt must be batched")


def smoke_test_vl_compute_loss(config: _config.TrainConfig):
    loader = _data_loader.create_vl_data_loader(
        config,
        shuffle=False,
        num_workers=0,
    )
    obs, actions = next(iter(loader))

    model = config.model.create(jax.random.key(0))
    loss, info = model.compute_loss(jax.random.key(1), obs, actions, train=False)

    loss_np = np.asarray(loss)
    _assert(np.all(np.isfinite(loss_np)), "VL compute_loss produced non-finite loss")


def smoke_test_cotrain_step(config: _config.TrainConfig, num_steps: int = 100):
    mesh = sharding.make_mesh(config.fsdp_devices)

    vla_loader, _ = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_workers=0,
    )
    vl_loader = _data_loader.create_vl_data_loader(
        config,
        shuffle=False,
        num_workers=0,
    )

    vla_iter = iter(vla_loader)
    vl_iter = iter(vl_loader)

    train_state, _ = init_train_state(config, jax.random.key(0), mesh, resume=False)
    state = train_state

    start_time = time.perf_counter()

    for i in range(num_steps):
        try:
            vla_batch = next(vla_iter)
        except StopIteration:
            vla_iter = iter(vla_loader)
            vla_batch = next(vla_iter)

        try:
            vl_batch = next(vl_iter)
        except StopIteration:
            vl_iter = iter(vl_loader)
            vl_batch = next(vl_iter)

        step_start = time.perf_counter()

        step_rng = jax.random.key(i)
        step_rng, vla_rng, vl_rng = jax.random.split(step_rng, 3)

        vla_grads, vla_loss, vla_info = compute_grad_vla(config, vla_rng, state, vla_batch)
        vl_grads, vl_loss, vl_info = compute_grad_vl(config, vl_rng, state, vl_batch)
        state, info = apply_combined_update(
            config,
            state,
            vla_grads, vl_grads,
            vla_loss, vl_loss,
            vla_info, vl_info,
        )

        # 等待 JAX 真正执行完，不然计时不准
        jax.block_until_ready(state.step)

        step = int(state.step)
        vla_loss = float(np.asarray(info["vla/loss"]))
        vl_loss = float(np.asarray(info["vl/loss"]))
        step_time = time.perf_counter() - step_start

        print(
            f"step={step:03d} "
            f"vla_loss={vla_loss:.6f} "
            f"vl_loss={vl_loss:.6f} "
            f"step_time={step_time:.4f}s"
        )

    total_time = time.perf_counter() - start_time
    avg_step_time = total_time / num_steps

    print(f"total_time={total_time:.4f}s")
    print(f"avg_step_time={avg_step_time:.4f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pifast_vlabench_cotrain")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument(
        "--only-data",
        action="store_true",
        help="Only run dataset/transform/loader tests; skip loss/train-step tests.",
    )
    args = parser.parse_args()

    config = _config.get_config(args.config)
    config = _make_smoke_config(config)

    # 基础检查，静默跑
    smoke_test_vl_dataset(config)
    smoke_test_vl_transform(config)
    smoke_test_vla_loader(config)
    smoke_test_vl_loader(config)

    if not args.only_data:
        smoke_test_vl_compute_loss(config)
        smoke_test_cotrain_step(config, num_steps=args.num_steps)


if __name__ == "__main__":
    main()