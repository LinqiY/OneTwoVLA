from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias
import threading
import time

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
try:
    # only used for onboard testing
    from pynput import keyboard
except ImportError:
    pass
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_fuse import Pi0Fuse
from openpi.models.pi0_fast_fuse import Pi0FASTFuse
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)[0],
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class ReasoningPolicy(BasePolicy):
    def __init__(
        self,
        model: Pi0Fuse,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        use_ref_img: bool = False,
        initial_scene_plan: str = '',
    ):
        self._prefill = nnx_utils.module_jit(model.prefill,
                                             static_argnames=('temprature',))
        self._act = nnx_utils.module_jit(model.act)
        self._reason = nnx_utils.module_jit(model.reason,
                                           static_argnames=('temprature',))
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._is_thinking = False
        self._lock = threading.Lock()

        self._use_ref_img = use_ref_img

        self._thought: str | None = None
        self._ref_img: np.ndarray | None = None
        self._initial_scene_plan: str = initial_scene_plan
        self._scene_plan: str | None = self._initial_scene_plan

        self._action_placehoser = np.zeros((model.action_horizon, model.action_dim))

        self._instruction: str | None = None

        self._intervention: str | None = None
        self._intervention_added: bool = False
        self._intervention_thread = threading.Thread(
            target=self._monitor_intervention, daemon=True)
        # self._intervention_thread.start()
        
        self._robot_question: str | None = None
        self._user_answer: str | None = None

        self._temperature = 0
        self._temperature = float(self._temperature)

    def start(self):
        """
        Start a new rollout.
        1. Reset the thinking flag.
        2. Reset the thought and ref_img.
        """
        self._reset_thought()

        print("Start a new rollout.")
    
    def _reset_thought(self):
        self.is_thinking = False
        self._thought = None
        self._ref_img = None
        self._scene_plan = self._initial_scene_plan
        self._instruction = None
        self._intervention = None
        self._intervention_added = False
        self._robot_question = None
        self._user_answer = None
    
    def _monitor_intervention(self):
        """
        Listen to the intervention signal from the user.

        If 'i' key is pressed, add the intervention.
        """
        def on_press(key):
            try:
                # When the 'i' key is pressed, add the intervention
                if key.char == 'i' and self._intervention is None:
                    self._intervention = "User Intervention: I don't want the orange-flavored vodka. Please Add the lemon-flavored vodka instead.\n"
                    self._intervention_added = True
                    print("\n'I' key pressed! Adding intervention...")
                    _intervention = self._intervention
                    _robot_question = '' if self._robot_question is None else self._robot_question
                    _user_answer = '' if self._user_answer is None else self._user_answer
                    self._thought = (self._instruction + 
                                    _robot_question + _user_answer + _intervention + self._scene_plan)
                    print('=============Thought after intervention=============')
                    print(self._thought)
                    print('-----------------------------------\n')
                    return False
            except AttributeError:
                # For non-character keys (like 'Esc', 'Shift', etc.),
                # we don't do anything.
                pass
        while True:
            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()
                time.sleep(2)

    def _prepare_obs(self, obs: dict) -> dict:
        """
        Update the `obs` dict's 
        'thought' and 'ref_img' fields.
        """
        assert 'prompt' in obs, \
            "The observation must contain a 'prompt' key."
        instruction = obs['prompt']
        
        if self._thought is None:
            self._instruction = 'Instruction: ' + instruction + '\n'
            self._thought = self._instruction + self._scene_plan
        obs['thought'] = [self._thought]
        
        if self._ref_img is None:
            assert 'image_1' in obs, \
                "The observation must contain an 'image_1' key."
            self._ref_img = obs['image_1']

        if self._use_ref_img:
            obs['reference_image'] = self._ref_img
        
        obs['act_with_outdated_thought'] = False
        obs['think_with_outdated_thought'] = False
        return obs

    def _update_thought(self, model_output: str, obs: dict):
        """
        Update the thought and ref_img fields.
        """
        _is_question = 'Robot Question:' in model_output
        if not _is_question:
            self._scene_plan = model_output
        if (self._robot_question is None and 
            'Robot Question' in model_output):
            self._robot_question = 'Robot Question: Which cocktail would you like? We have three options: Mojito, Mount Fuji, and Vodka Sunrise.\n'
            self._user_answer = 'User Answer: I want a cup of Vodka Sunrise.\n'
            # self._user_answer = 'User Answer: I want a cup of Mojito.\n'
            # self._user_answer = 'User Answer: I want a cup of Mount Fuji.\n'
        _intervention = '' if self._intervention is None else self._intervention
        _robot_question = '' if self._robot_question is None else self._robot_question
        _user_answer = '' if self._user_answer is None else self._user_answer
        self._thought = (self._instruction + 
                         _robot_question + _user_answer + _intervention + self._scene_plan)

        self._ref_img = obs['image_1']
        print('Updated prefix: ')
        print(self._thought)
        print('-----------------------------------\n')
    
    def _warmup(self, inputs: dict) -> dict:
        """warmup jit compilation"""
        print('Warmup...')
        inputs = self._prepare_obs(inputs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        inputs = _model.FuseObservation.from_dict(inputs)

        prefill_rng, think_rng, act_rng, self._rng = jax.random.split(self._rng, 4)

        # prefill and decide to act or think
        processed_inputs, prefix_cache, first_suffix_token, \
            last_logit, prefix_mask, prefix_positions, to_act =  self._prefill(prefill_rng, inputs)
        
        # forward both think and act
        # we keep is_thinking False here to block the policy serving
        actions = self._act(act_rng, processed_inputs, prefix_cache,
                            prefix_mask, prefix_positions)
        suffix_tokens = self._reason(think_rng, last_logit,
                                    prefix_cache, prefix_mask, prefix_positions)
        
        outputs = {
            "state": inputs.state,
            "actions": actions,
            "tokenized_suffix": suffix_tokens,
        }

        self._reset_thought()
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        transformed = self._output_transform(outputs)
        return transformed

    @override
    def infer(self, obs: dict) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        warmup = inputs.pop('is_warm_up', False)
        if warmup:
            return self._warmup(inputs)

        inputs = self._prepare_obs(inputs)
        prefix = inputs['thought'][0]
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        inputs = _model.FuseObservation.from_dict(inputs)

        prefill_rng, think_rng, act_rng, self._rng = jax.random.split(self._rng, 4)

        # prefill and decide to act or think
        (
            processed_inputs,
            prefix_cache,
            first_suffix_token,
            last_logit,
            prefix_mask,
            prefix_positions,
            to_act
         ) =  self._prefill(prefill_rng, inputs, temprature=self._temperature)
        assert jnp.size(to_act) == 1
        to_act = to_act.item()
        to_think = not to_act

        if self._scene_plan == self._initial_scene_plan:
            to_think = True
            to_act = False

        # update is_thinking
        self.is_thinking = to_think

        # defaults:
        actions = self._action_placehoser[np.newaxis, ...]
        # default: '<eos>' token
        suffix_tokens = np.ones((1, 1), dtype=np.int32)
        if to_act:
            actions = self._act(act_rng, processed_inputs,
                                prefix_cache, prefix_mask, prefix_positions)
        else:
            print('Prefix:')
            print(prefix)
            print('Thinking...')
            suffix_tokens = self._reason(think_rng, last_logit, prefix_cache, prefix_mask,
                                        prefix_positions, temprature=self._temperature)
        outputs = {
            "state": inputs.state,
            "actions": actions,
            "tokenized_suffix": suffix_tokens,
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        transformed = self._output_transform(outputs)

        # update thought and ref_img
        if to_think:
            print(transformed['thoughts'])
            self._update_thought(transformed['thoughts'], obs)
            # thinking is done 
            self.is_thinking = False
            return {'isthinking': np.True_}

        return transformed
    
    @property
    def is_thinking(self) -> bool:
        with self._lock:
            return self._is_thinking
    
    @is_thinking.setter
    def is_thinking(self, value: bool):
        with self._lock:
            self._is_thinking = value

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class FASTReasoningPolicy(BasePolicy):
    """Reasoning policy for Pi0FASTFuse: both reasoning and action are autoregressive token decoding."""

    def __init__(
        self,
        model: Pi0FASTFuse,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        use_ref_img: bool = False,
        initial_scene_plan: str = '',
    ):
        self._prefill = nnx_utils.module_jit(model.prefill, static_argnames=('temperature',))
        self._act = nnx_utils.module_jit(model.act)
        self._reason = nnx_utils.module_jit(model.reason, static_argnames=('temperature',))
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._is_thinking = False
        self._lock = threading.Lock()

        self._use_ref_img = use_ref_img

        self._thought: str | None = None
        self._ref_img: np.ndarray | None = None
        self._initial_scene_plan: str = initial_scene_plan
        self._scene_plan: str | None = self._initial_scene_plan

        self._action_horizon = model.action_horizon
        self._action_dim = model.action_dim

        self._instruction: str | None = None

        self._temperature = 0.0

    def start(self):
        self._reset_thought()
        print("Start a new rollout.")

    def _reset_thought(self):
        self.is_thinking = False
        self._thought = None
        self._ref_img = None
        self._scene_plan = self._initial_scene_plan
        self._instruction = None

    def _prepare_obs(self, obs: dict) -> dict:
        assert 'prompt' in obs, "The observation must contain a 'prompt' key."
        instruction = obs['prompt']

        if self._thought is None:
            self._instruction = 'Instruction: ' + instruction + '\n'
            self._thought = self._instruction + self._scene_plan
        obs['thought'] = [self._thought]

        if self._ref_img is None:
            assert 'image_1' in obs, "The observation must contain an 'image_1' key."
            self._ref_img = obs['image_1']

        if self._use_ref_img:
            obs['reference_image'] = self._ref_img

        obs['act_with_outdated_thought'] = False
        obs['think_with_outdated_thought'] = False
        return obs

    def _update_thought(self, model_output: str, obs: dict):
        self._scene_plan = model_output
        self._thought = self._instruction + self._scene_plan
        self._ref_img = obs['image_1']
        print('Updated prefix: ')
        print(self._thought)
        print('-----------------------------------\n')

    def _warmup(self, inputs: dict) -> dict:
        print('Warmup...')
        inputs = self._prepare_obs(inputs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        inputs = _model.FuseObservation.from_dict(inputs)

        prefill_rng, think_rng, act_rng, self._rng = jax.random.split(self._rng, 4)

        processed_inputs, prefix_cache, first_suffix_token, \
            last_logit, prefix_mask, prefill_len, prefill_size, to_act = \
            self._prefill(prefill_rng, inputs)

        action_tokens = self._act(act_rng, prefix_cache, prefix_mask,
                                  prefill_len, prefill_size)
        suffix_tokens = self._reason(think_rng, last_logit, prefix_cache,
                                     prefill_len, prefill_size)

        outputs = {
            "state": inputs.state,
            "actions": action_tokens,
            "tokenized_suffix": suffix_tokens,
        }

        self._reset_thought()
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        transformed = self._output_transform(outputs)
        return transformed

    @override
    def infer(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        warmup = inputs.pop('is_warm_up', False)
        if warmup:
            return self._warmup(inputs)

        inputs = self._prepare_obs(inputs)
        prefix = inputs['thought'][0]
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        inputs = _model.FuseObservation.from_dict(inputs)

        prefill_rng, think_rng, act_rng, self._rng = jax.random.split(self._rng, 4)

        (
            processed_inputs,
            prefix_cache,
            first_suffix_token,
            last_logit,
            prefix_mask,
            prefill_len,
            prefill_size,
            to_act,
        ) = self._prefill(prefill_rng, inputs, temperature=self._temperature)
        assert jnp.size(to_act) == 1
        to_act = to_act.item()
        to_think = not to_act

        if self._scene_plan == self._initial_scene_plan:
            to_think = True
            to_act = False

        self.is_thinking = to_think

        action_tokens = np.zeros((1, 256), dtype=np.int32)
        suffix_tokens = np.ones((1, 1), dtype=np.int32)

        if to_act:
            action_tokens = self._act(act_rng, prefix_cache, prefix_mask,
                                      prefill_len, prefill_size)
        else:
            print('Prefix:')
            print(prefix)
            print('Thinking...')
            suffix_tokens = self._reason(think_rng, last_logit, prefix_cache,
                                         prefill_len, prefill_size,
                                         temperature=self._temperature)

        outputs = {
            "state": inputs.state,
            "actions": action_tokens,
            "tokenized_suffix": suffix_tokens,
        }

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        transformed = self._output_transform(outputs)

        if to_think:
            print(transformed['thoughts'])
            self._update_thought(transformed['thoughts'], obs)
            self.is_thinking = False
            return {'isthinking': np.True_}

        return transformed

    @property
    def is_thinking(self) -> bool:
        with self._lock:
            return self._is_thinking

    @is_thinking.setter
    def is_thinking(self, value: bool):
        with self._lock:
            self._is_thinking = value

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
