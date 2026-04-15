# scripts/train_cotrain.py

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
from flax.training import common_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
import wandb


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        wandb_name = config.exp_name + "-resumed"
        wandb.init(
            name=wandb_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _prefix_metrics(prefix: str, metrics: dict[str, at.Array]) -> dict[str, at.Array]:
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


# ─────────────────────────────────────────────
# 拆分后的三个 jit 函数
# ─────────────────────────────────────────────

def compute_grad_vla(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    vla_batch: tuple[_model.Observation, _model.Actions],
):
    """VLA forward/backward，返回梯度、loss 和 info。"""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model_, rng_, observation_, actions_):
        chunked_loss, info = model_.compute_loss(rng_, observation_, actions_, train=True)
        return jnp.mean(chunked_loss), info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    vla_observation, vla_actions = vla_batch
    (vla_loss, vla_info), vla_grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, rng, vla_observation, vla_actions
    )
    return vla_grads, vla_loss, vla_info


def compute_grad_vl(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    vl_batch: tuple[_model.Observation, at.Array],
):
    """VL forward/backward，返回梯度、loss 和 info。"""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model_, rng_, observation_, actions_):
        chunked_loss, info = model_.compute_loss(rng_, observation_, actions_, train=True)
        return jnp.mean(chunked_loss), info

    diff_state = nnx.DiffState(0, config.trainable_filter)
    vl_observation, vl_actions = vl_batch
    (vl_loss, vl_info), vl_grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, rng, vl_observation, vl_actions
    )
    return vl_grads, vl_loss, vl_info


def apply_combined_update(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    vla_grads,
    vl_grads,
    vla_loss,
    vl_loss,
    vla_info,
    vl_info,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """合并梯度，执行一次 optimizer update，返回新 state 和 metrics。"""
    combined_grads = jax.tree.map(lambda a, b: a + b, vla_grads, vl_grads)

    trainable_params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(combined_grads, state.opt_state, trainable_params)
    new_trainable_params = optax.apply_updates(trainable_params, updates)

    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, new_trainable_params)
    final_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=final_params,
        opt_state=new_opt_state,
    )

    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                final_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        "loss": vla_loss + vl_loss,
        "param_norm": optax.global_norm(kernel_params),
        "vla/grad_norm": optax.global_norm(vla_grads),
        "vl/grad_norm": optax.global_norm(vl_grads),
        "combined/grad_norm": optax.global_norm(combined_grads),
    }
    info.update(_prefix_metrics("vla", {"loss": vla_loss, **vla_info}))
    info.update(_prefix_metrics("vl", {"loss": vl_loss, **vl_info}))
    return new_state, info


def _create_output_transform(config: _config.TrainConfig) -> tuple[_transforms.DataTransformFn, _transforms.DataTransformFn]:
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = data_config.norm_stats
    output_transform = _transforms.CompositeTransform(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    _output_transform_list = output_transform.transforms
    _target_transform_list = [x for x in _output_transform_list if not isinstance(x, _transforms.ExtractFASTActions)]
    target_transform = _transforms.CompositeTransform(_target_transform_list)
    return output_transform, target_transform


@at.typecheck
def infer_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation | _model.FuseObservation, _model.Actions],
) -> dict[str, at.Array]:
    if state.ema_decay is None:
        model = nnx.merge(state.model_def, state.params)
    else:
        model = nnx.merge(state.model_def, state.ema_params)
    model.eval()

    sample_actions = nnx_utils.module_jit(model.sample_actions)

    infer_rng = jax.random.fold_in(rng, state.step)
    observation, targets = batch
    actions, val_info = sample_actions(rng=infer_rng, observation=observation)

    _inputs = jax.tree.map(lambda x: x, observation)
    _targets = jax.tree.map(lambda x: x, targets)
    _action_mask = jnp.ones(actions.shape[0], dtype=jnp.bool_)
    if isinstance(observation, _model.FuseObservation):
        _action_mask = _inputs.diffusion_loss_mask

    outputs = {
        "state": _inputs.state,
        "actions": actions,
        "targets": _targets,
        "action_mask": _action_mask,
        **val_info,
    }
    return outputs


@at.typecheck
def compute_mse(
    state: at.Float[at.Array, "b s"],
    actions: at.Float[at.Array, "b ah ad"],
    targets: at.Float[at.Array, "b ah ad"],
    action_mask: at.Bool[at.Array, "b"],
    output_transform: _transforms.DataTransformFn,
    target_transform: _transforms.DataTransformFn,
) -> dict[str, at.ArrayLike]:
    batch_size = state.shape[0]
    errors = []
    for i in range(batch_size):
        state_i = np.asarray(state[i])
        action_i = np.asarray(actions[i])
        target_i = np.asarray(targets[i])
        transformed_action_i = output_transform({"state": state_i, "actions": action_i})["actions"]
        transformed_target_i = target_transform({"state": state_i, "actions": target_i})["actions"]
        errors.append(transformed_target_i - transformed_action_i)

    errors = np.asarray(errors)
    broadcasted_action_mask = np.broadcast_to(action_mask[:, None, None], errors.shape)
    if np.sum(broadcasted_action_mask) == 0:
        mse = np.nan
    else:
        mse = np.mean(errors[broadcasted_action_mask] ** 2)
    return {
        "action_mse": mse,
        "num_action_loss_fraction": np.sum(action_mask) / batch_size,
    }


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    if not hasattr(config, "vl_data"):
        raise ValueError("train_cotrain.py expects a config with `vl_data`.")
    if not hasattr(config, "cotrain_ratio"):
        raise ValueError("train_cotrain.py expects a config with `cotrain_ratio`.")

    # jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("/inspire/hdd/global_user/gongjingjing-25039/lqyin/OneTwoVLA/jax_compilation_cache_dir").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    if config.use_val_dataset:
        val_rng, _ = jax.random.split(train_rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # ── VLA data loader ──────────────────────────────────────────────────────
    data_loader, val_data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    vla_batch = next(data_iter)
    logging.info(f"Initialized VLA data loader:\n{training_utils.array_tree_to_info(vla_batch)}")
    training_utils.inspect_prompts(vla_batch)

    # ── VL data loader ───────────────────────────────────────────────────────
    vl_data_loader = _data_loader.create_vl_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    vl_data_iter = iter(vl_data_loader)
    vl_batch = next(vl_data_iter)
    logging.info(f"Initialized VL data loader:\n{training_utils.array_tree_to_info(vl_batch)}")

    if config.use_val_dataset and val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
        val_batch = next(val_data_iter)
        logging.info(f"Initialized validation data loader:\n{training_utils.array_tree_to_info(val_batch)}")
    else:
        val_data_iter = None

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # ── 三个独立 jit，顺序执行以节省显存 ────────────────────────────────────
    pvla_grad = jax.jit(
        functools.partial(compute_grad_vla, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    pvl_grad = jax.jit(
        functools.partial(compute_grad_vl, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    papply_update = jax.jit(
        functools.partial(apply_combined_update, config),
        in_shardings=(
            train_state_sharding,  # state
            replicated_sharding,   # vla_grads
            replicated_sharding,   # vl_grads
            replicated_sharding,   # vla_loss
            replicated_sharding,   # vl_loss
            replicated_sharding,   # vla_info
            replicated_sharding,   # vl_info
        ),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(0,),
    )

    pval_inference = None
    if config.use_val_dataset and val_data_loader is not None:
        pval_inference = jax.jit(
            functools.partial(infer_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        output_transform, target_transform = _create_output_transform(config)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # 每个 step 独立拆分 rng，和原来保持一致
        step_rng = jax.random.fold_in(train_rng, step)
        step_rng, vla_rng, vl_rng = jax.random.split(step_rng, 3)

        with sharding.set_mesh(mesh):
            # 顺序执行：VLA grad → VL grad（激活各自释放）→ 合并更新
            vla_grads, vla_loss, vla_info = pvla_grad(vla_rng, train_state, vla_batch)
            vl_grads, vl_loss, vl_info = pvl_grad(vl_rng, train_state, vl_batch)
            train_state, info = papply_update(
                train_state,
                vla_grads, vl_grads,
                vla_loss, vl_loss,
                vla_info, vl_info,
            )
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Train at step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if (step % config.val_interval == 0 or step == start_step) and config.use_val_dataset and pval_inference is not None:
            val_infos = []
            for val_batch in tqdm.tqdm(val_data_loader, dynamic_ncols=True, desc="Validation", leave=False):
                with sharding.set_mesh(mesh):
                    val_outputs = pval_inference(val_rng, train_state, val_batch)
                val_info = compute_mse(
                    state=val_outputs["state"],
                    actions=val_outputs["actions"],
                    targets=val_outputs["targets"],
                    action_mask=val_outputs["action_mask"],
                    output_transform=output_transform,
                    target_transform=target_transform,
                )
                if "text_loss" in val_outputs:
                    val_info["text_loss"] = val_outputs["text_loss"]
                val_infos.append(val_info)

            stacked_val_infos = common_utils.stack_forest(val_infos)
            reduced_val_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_val_infos))
            val_info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_val_info.items())
            pbar.write(f"Validation at step {step}: {val_info_str}")
            wandb.log({f"val/{k}": v for k, v in reduced_val_info.items()}, step=step)

        # Next VLA batch
        try:
            vla_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            vla_batch = next(data_iter)

        # Next VL batch
        try:
            vl_batch = next(vl_data_iter)
        except StopIteration:
            vl_data_iter = iter(vl_data_loader)
            vl_batch = next(vl_data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())