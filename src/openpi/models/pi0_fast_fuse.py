import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
import openpi.models.tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1

from openpi.models.pi0_fast import make_attn_mask, left_to_right_align, put_along_last_axis


@dataclasses.dataclass(frozen=True)
class Pi0FASTFuseConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 400

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST_FUSE

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FASTFuse":
        return Pi0FASTFuse(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.FuseObservation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.FuseObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                diffusion_loss_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FASTFuse(_model.BaseModel):
    """Pi0-FAST with fused reasoning: all generation (reasoning + FAST action tokens) is autoregressive."""

    def __init__(self, config: Pi0FASTFuseConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

    @at.typecheck
    def embed_inputs(
        self, obs: _model.FuseObservation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        token_embeddings = []

        for name in obs.images:
            image_emb, _ = self.PaliGemma.img(obs.images[name], train=False)
            token_embeddings.append(image_emb)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_emb.shape[1])
            )
            ar_mask.append(0 * input_mask[-1])

        assert obs.tokenized_prompt is not None
        assert obs.tokenized_prompt_mask is not None
        assert obs.token_ar_mask is not None

        txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(txt_emb)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.FuseObservation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        emb, mask, ar = self.embed_inputs(observation)
        attn_mask = make_attn_mask(mask, ar)

        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=emb[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1]:],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        assert observation.token_loss_mask is not None
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

        info = {
            "num_action_loss_fraction": (
                jnp.sum(observation.diffusion_loss_mask)
                / observation.diffusion_loss_mask.shape[0]
            ),
        }
        return loss, info

    @at.typecheck
    def prefill(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        *,
        temperature: float = 0.0,
    ):
        """Prefill the prefix and decide whether to act or reason.

        Returns:
            observation, kv_cache, token, eop_logit,
            prefix_mask, prefill_len, prefill_size, has_boa
        """
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        first_one = jnp.argmax(observation.token_ar_mask, axis=-1)
        padding = jnp.arange(observation.token_ar_mask.shape[-1]) >= first_one[..., jnp.newaxis]
        observation = dataclasses.replace(
            observation,
            tokenized_prompt=jnp.where(padding, 0, observation.tokenized_prompt),
            tokenized_prompt_mask=jnp.logical_not(padding),
        )

        emb, mask, ar = self.embed_inputs(observation)
        attn_mask = make_attn_mask(mask, ar)

        emb, mask, attn_mask = left_to_right_align(emb, mask, attn_mask)
        prefill_size = emb.shape[1]
        prefill_len = jnp.sum(mask, axis=-1)

        max_decode = 512
        attn_mask_padded = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, max_decode)))
        positions = jnp.cumsum(mask, axis=-1) - 1

        logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=emb, mask=attn_mask_padded, positions=positions, decode=True
        )
        last_logit = logits[:, -1:]

        valid_tokens = jnp.array([_tokenizer.BEGIN_OF_ACTION, _tokenizer.BEGIN_OF_REASONING])
        valid_mask = jnp.full((1, 1, last_logit.shape[-1]), -jnp.inf)
        valid_mask = valid_mask.at[:, :, valid_tokens].set(0)
        eop_logit = last_logit + valid_mask

        if temperature > 0.0:
            token = jax.random.categorical(rng, eop_logit / temperature, axis=-1)
        else:
            token = jnp.argmax(eop_logit, axis=-1)

        has_boa = jnp.any(token == _tokenizer.BEGIN_OF_ACTION, axis=1)

        return observation, kv_cache, token, eop_logit, mask, prefill_len, prefill_size, has_boa

    @at.typecheck
    def reason(
        self,
        rng: at.KeyArrayLike,
        last_logit: at.Float[at.Array, "b 1 v"],
        kv_cache,
        prefill_len: at.Int[at.Array, " b"],
        prefill_size: int,
        *,
        temperature: float = 0.0,
        max_decoding_steps: int = 256,
    ) -> at.Int[at.Array, "b _s"]:
        """Autoregressively decode reasoning tokens until EOS."""
        prefix_start = prefill_size - prefill_len
        cache_size = prefill_size + max_decoding_steps

        if temperature > 0.0:
            token = jax.random.categorical(rng, last_logit / temperature, axis=-1)
        else:
            token = jnp.argmax(last_logit, axis=-1)

        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        def decode_step(carry):
            last_logit, output_tokens, cache, _, step = carry
            step_rng = jax.random.fold_in(rng, step)
            if temperature > 0.0:
                tok = jax.random.categorical(step_rng, last_logit / temperature, axis=-1)
            else:
                tok = jnp.argmax(last_logit, axis=-1)
            tok = jnp.where(
                step == 0,
                jnp.full_like(tok, _tokenizer.BEGIN_OF_REASONING),
                tok,
            )
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (tok.shape[0], 1)), tok
            )
            has_eos = jnp.all(jnp.any(tok == PALIGEMMA_EOS_TOKEN, axis=-1))

            tok_emb = self.PaliGemma.llm(tok, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(cache_size)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(cache_size)[None, None, :] < (prefill_size + step + 1),
            )
            new_logit, new_cache, _ = self.PaliGemma.llm(
                embedded_prefix=tok_emb, mask=mask, positions=positions, decode=True, kv_cache=cache
            )
            return new_logit[:, -1:], output_tokens, new_cache, has_eos, step + 1

        def decode_cond(carry):
            _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        _, suffix_tokens, _, _, _ = jax.lax.while_loop(
            decode_cond, decode_step,
            (last_logit, output_tokens, kv_cache, False, 0),
        )
        return suffix_tokens

    @at.typecheck
    def act(
        self,
        rng: at.KeyArrayLike,
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefill_len: at.Int[at.Array, " b"],
        prefill_size: int,
        *,
        max_decoding_steps: int = 256,
        temperature: float = 0.0,
    ) -> at.Int[at.Array, "b _s"]:
        """Autoregressively decode FAST action tokens after prefill."""
        prefix_start = prefill_size - prefill_len
        cache_size = prefill_size + max_decoding_steps
        batch_size = prefill_len.shape[0]

        boa = jnp.broadcast_to(_tokenizer.BEGIN_OF_ACTION, (batch_size, 1))
        boa_emb = self.PaliGemma.llm(boa, embed_only=True)
        positions = prefill_len[:, None] + 1
        mask = jnp.logical_and(
            jnp.arange(cache_size)[None, None, :] >= prefix_start[:, None, None],
            jnp.arange(cache_size)[None, None, :] < (prefill_size + 1),
        )
        last_logit, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=boa_emb, mask=mask, positions=positions, decode=True, kv_cache=kv_cache
        )
        last_logit = last_logit[:, -1:]

        output_tokens = jnp.zeros((batch_size, max_decoding_steps), dtype=jnp.int32)

        def step_fn(carry):
            last_logit, out, cache, _, step_i = carry
            if temperature > 0.0:
                tok = jax.random.categorical(rng, last_logit / temperature, axis=-1)
            else:
                tok = jnp.argmax(last_logit, axis=-1)
            out = put_along_last_axis(out, jnp.broadcast_to(step_i, (tok.shape[0], 1)), tok)
            has_eos = jnp.all(jnp.any(tok == PALIGEMMA_EOS_TOKEN, axis=-1))

            tok_emb = self.PaliGemma.llm(tok, embed_only=True)
            pos = prefill_len[:, None] + step_i + 2
            m = jnp.logical_and(
                jnp.arange(cache_size)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(cache_size)[None, None, :] < (prefill_size + step_i + 2),
            )
            new_logit, new_cache, _ = self.PaliGemma.llm(
                embedded_prefix=tok_emb, mask=m, positions=pos, decode=True, kv_cache=cache
            )
            return new_logit[:, -1:], out, new_cache, has_eos, step_i + 1

        def cond_fn(carry):
            _, _, _, all_eos, step_i = carry
            return (~all_eos) & (step_i < max_decoding_steps)

        _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond_fn, step_fn, (last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        """Full forward pass for validation: embed everything and decode autoregressively."""
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        emb, mask, ar = self.embed_inputs(observation)
        attn_mask = make_attn_mask(mask, ar)

        emb, mask, attn_mask = left_to_right_align(emb, mask, attn_mask)
        prefill_size = emb.shape[1]
        prefill_len = jnp.sum(mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        attn_mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        positions = jnp.cumsum(mask, axis=-1) - 1
        logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=emb, mask=attn_mask, positions=positions, decode=True
        )

        last_logit = logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        def step_fn(carry):
            last_logit, out, cache, _, s = carry
            if temperature > 0.0:
                tok = jax.random.categorical(rng, last_logit / temperature, axis=-1)
            else:
                tok = jnp.argmax(last_logit, axis=-1)
            out = put_along_last_axis(out, jnp.broadcast_to(s, (tok.shape[0], 1)), tok)
            has_eos = jnp.all(jnp.any(tok == PALIGEMMA_EOS_TOKEN, axis=-1))
            tok_emb = self.PaliGemma.llm(tok, embed_only=True)
            pos = prefill_len[:, None] + s + 1
            m = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] < (prefill_size + s + 1),
            )
            new_logit, new_cache, _ = self.PaliGemma.llm(
                embedded_prefix=tok_emb, mask=m, positions=pos, decode=True, kv_cache=cache
            )
            return new_logit[:, -1:], out, new_cache, has_eos, s + 1

        def cond_fn(carry):
            _, _, _, all_eos, s = carry
            return (~all_eos) & (s < max_decoding_steps)

        _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond_fn, step_fn, (last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens, {}
