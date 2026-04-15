from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
import openpi.policies.umi_dataset as umi_dataset

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    *,
    use_val_dataset: bool = False,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """Create datasets for training. UMI uses episode-based split; LeRobot uses random frame split when use_val_dataset."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, local_files_only=data_config.local_files_only)

    val_dataset = None
    if isinstance(data_config, _config.UMIDataConfig):
        dataset = umi_dataset.UMIDataset(data_config, model_config.action_horizon)
        if data_config.use_val_dataset:
            val_dataset = dataset.get_val_dataset()
    else:
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
                for key in data_config.action_sequence_keys
            },
            local_files_only=data_config.local_files_only,
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        if val_dataset is not None:
            val_dataset = TransformedDataset(val_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    if use_val_dataset and val_dataset is None and not isinstance(data_config, _config.UMIDataConfig):
        n = len(dataset)
        if n > 1:
            n_val = min(max(1, int(n * val_ratio)), n - 1)
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            val_indices = perm[:n_val].tolist()
            train_indices = perm[n_val:].tolist()
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            dataset = torch.utils.data.Subset(dataset, train_indices)

    return dataset, val_dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader[tuple[_model.Observation, _model.Actions]],\
            DataLoader[tuple[_model.Observation, _model.Actions]] | None]:
    """Create data loaders for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset, val_dataset = create_dataset(
        data_config,
        config.model,
        use_val_dataset=config.use_val_dataset,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    if val_dataset is not None:
        val_dataset = transform_dataset(val_dataset, data_config.create_val_config(), skip_norm_stats=skip_norm_stats)
        val_data_loader = TorchDataLoader(
            val_dataset,
            local_batch_size=config.val_batch_size // jax.process_count(),
            sharding=sharding,
            shuffle=False,
            num_batches=num_batches,
            num_workers=num_workers,
            iterate_indefinitely=False, # Don't iterate indefinitely for validation.
            seed=config.seed,
        )
    else:
        val_data_loader = None

    class DataLoaderImpl(DataLoader):
        def __init__(
            self,
            data_config: _config.DataConfig,
            data_loader: TorchDataLoader,
            model_config: _model.BaseModelConfig,
        ):
            self._data_config = data_config
            self._data_loader = data_loader
            self._model_config = model_config

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                use_fuse_obs = (
                    hasattr(self._data_config, "use_reasoning") and self._data_config.use_reasoning
                ) or self._model_config.model_type in (_model.ModelType.PI0_FUSE, _model.ModelType.PI0_FAST_FUSE)
                if use_fuse_obs:
                    yield _model.FuseObservation.from_dict(batch), batch["actions"]
                else:
                    yield _model.Observation.from_dict(batch), batch["actions"]
        
        def __len__(self) -> int:
            return len(self._data_loader)

    return DataLoaderImpl(data_config, data_loader, config.model), \
        DataLoaderImpl(data_config, val_data_loader, config.model) if val_data_loader else None

def create_vl_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a VL/VQA data loader for co-training."""
    from openpi.policies.vl_dataset import ShareRobotVQADataset

    vl_data_config = config.vl_data.create(config.assets_dirs, config.model)

    if not vl_data_config.vl_groups:
        raise ValueError("VLDataConfig.vl_groups is empty after create(); check vl_data factory settings.")

    per_group_ds = [
        ShareRobotVQADataset(json_paths=g.json_paths, image_root=g.image_root) for g in vl_data_config.vl_groups
    ]
    for d, g in zip(per_group_ds, vl_data_config.vl_groups, strict=True):
        if len(d) == 0:
            raise ValueError(f"VL group has no samples after loading JSON paths (first paths): {g.json_paths[:3]}")

    if len(per_group_ds) == 1:
        raw_vl = per_group_ds[0]
    else:
        raw_vl = torch.utils.data.ConcatDataset(per_group_ds)

    dataset = transform_dataset(raw_vl, vl_data_config, skip_norm_stats=skip_norm_stats)

    use_shuffle = shuffle and vl_data_config.vl_shuffle
    sampler: torch.utils.data.Sampler | None = None
    if len(per_group_ds) > 1 and use_shuffle:
        lengths = [len(d) for d in per_group_ds]
        probs = np.array([g.interleave_prob for g in vl_data_config.vl_groups], dtype=np.float64)
        if np.any(probs < 0):
            raise ValueError("interleave_prob must be non-negative for every VL group.")
        psum = float(np.sum(probs))
        if psum <= 0:
            raise ValueError("Sum of interleave_prob over VL groups must be > 0.")
        probs = probs / psum
        weights: list[float] = []
        for n_i, p_i in zip(lengths, probs, strict=True):
            w = float(p_i) / float(n_i)
            weights.extend([w] * n_i)
        wgen = torch.Generator()
        wgen.manual_seed(int(config.seed))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=wgen,
        )
        use_shuffle = False

    global_vl_batch_size = max(1, round(config.batch_size * config.cotrain_ratio))
    num_devices = jax.device_count()

    if global_vl_batch_size < num_devices:
        raise ValueError(
            f"VL global batch size ({global_vl_batch_size}) is smaller than device count ({num_devices}). "
            f"Current batch_size={config.batch_size}, cotrain_ratio={config.cotrain_ratio}. "
            "Increase batch_size or cotrain_ratio."
        )
    if global_vl_batch_size % num_devices != 0:
        raise ValueError(
            f"VL global batch size ({global_vl_batch_size}) must be divisible by device count ({num_devices}). "
            f"Current batch_size={config.batch_size}, cotrain_ratio={config.cotrain_ratio}."
        )

    local_vl_batch_size = global_vl_batch_size // jax.process_count()

    vl_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_vl_batch_size,
        sharding=sharding,
        shuffle=use_shuffle,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(
            self,
            data_config: _config.DataConfig,
            data_loader: TorchDataLoader,
            model_config: _model.BaseModelConfig,
        ):
            self._data_config = data_config
            self._data_loader = data_loader
            self._model_config = model_config

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

        def __len__(self) -> int:
            return len(self._data_loader)

    return DataLoaderImpl(vl_data_config, vl_loader, config.model)

class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        iterate_indefinitely: bool = True,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data (ignored if `sampler` is set).
            sampler: Optional sampler (e.g. weighted interleave across concatenated VL pools).
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            iterate_indefinitely: Whether to iterate over the dataset indefinitely.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sampler is not None and shuffle:
            raise ValueError("Cannot use `shuffle=True` together with a custom `sampler`.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )
        self._iterate_indefinitely = iterate_indefinitely

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if self._iterate_indefinitely:
                        break  # We've exhausted the dataset. Create a new iterator and start over.
                    else:
                        return # We've exhausted the dataset and we're not iterating indefinitely.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
    
    def __len__(self) -> int:
        if self._iterate_indefinitely:
            raise ValueError("Cannot determine the length of an indefinitely iterating data loader.")
        return len(self._data_loader)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
