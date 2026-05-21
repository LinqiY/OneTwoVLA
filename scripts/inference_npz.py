#!/usr/bin/env python3
"""
用 OpenPI 加载 PaliGemma npz 做 VLM 推理（文字生成）
用法:
    python paligemma_infer_openpi.py <image_path> "<prompt>"
例子:
    python paligemma_infer_openpi.py test.jpg "what do you see"
"""

import sys
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
import sentencepiece as spm
import flax.nnx as nnx

from openpi.training import config as _config
from openpi.training import weight_loaders
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

# ── 路径配置 ──────────────────────────────────────────
NPZ_PATH       = "/inspire/hdd/global_user/gongjingjing-25039/lqyin/models/paligemma-3b-pt-224-jax/paligemma-3b-pt-224.npz"
TOKENIZER_PATH = "/inspire/hdd/global_user/gongjingjing-25039/lqyin/models/paligemma_tokenizer.model"
IMAGE_PATH     = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
PROMPT         = sys.argv[2] if len(sys.argv) > 2 else "describe the image"
MAX_NEW_TOKENS = 50
IMAGE_SIZE     = 224
BOS_TOKEN      = 2
EOS_TOKEN      = 1
# ─────────────────────────────────────────────────────


def load_model(npz_path):
    """加载 PaliGemma npz 权重到 OpenPI Pi0 模型结构"""
    config = _config.get_config("pifast_vlabench_pretrain_primitive")  # pifast_w_vlabench_pretrain_primitive_test
    model_config = config.model

    print("Initializing model structure...")
    dummy_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    _, state = nnx.split(dummy_model)
    empty_params = state.to_pure_dict()

    print(f"Loading weights from {npz_path}...")
    loader = weight_loaders.PaliGemmaWeightLoader(npz_path=npz_path)
    params = loader.load(empty_params)

    print("Creating model with loaded weights...")
    model = model_config.load(params)
    return model


def load_tokenizer(path):
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp


def preprocess_image(image_path):
    """加载图像 -> (1, H, W, 3) float32 in [-1, 1]"""
    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return img[None]  # (1, 224, 224, 3)


def build_observation(image, prompt_tokens, max_token_len=180):
    B = image.shape[0]

    tokens = [BOS_TOKEN] + prompt_tokens
    token_len = len(tokens)
    padded = tokens + [0] * (max_token_len - token_len)
    padded = padded[:max_token_len]

    obs = _model.Observation.from_dict({
        "image": {
            "base_0_rgb": jnp.array(image),                                    # numpy -> jnp
        },
        "image_mask": {
            "base_0_rgb": jnp.ones(B, dtype=jnp.bool_),                       # numpy -> jnp
        },
        "state": jnp.zeros((B, 7), dtype=jnp.float32),                        # numpy -> jnp
        "tokenized_prompt": jnp.array([padded], dtype=jnp.int32),             # numpy -> jnp
        "tokenized_prompt_mask": jnp.array(                                    # numpy -> jnp
            [[True] * token_len + [False] * (max_token_len - token_len)],
            dtype=jnp.bool_
        ),
    })
    return obs


def generate(model, obs, tokenizer, max_new_tokens=50):
    """
    用 PaliGemma LLM 做自回归生成。
    embed_prefix 把图像和 prompt 编码成 prefix tokens，
    然后用 LLM 自回归生成新 token。
    """
    print("Embedding prefix (image + prompt)...")

    # preprocess observation（归一化图像等）
    obs = _model.preprocess_observation(
        None, obs, train=False,
        image_keys=list(obs.images.keys())
    )

    # embed prefix: 图像 token + prompt token
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    print(f"Prefix tokens shape: {prefix_tokens.shape}")

    # 构造 attention mask 和 positions
    from openpi.models.pi0 import make_attn_mask
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    # 填充 KV cache
    print("Building KV cache...")
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=positions
    )

    # 自回归生成
    generated_tokens = []
    # 从 prefix 最后一个位置开始
    next_token = jnp.array([[BOS_TOKEN]], dtype=jnp.int32)
    prefix_len = int(jnp.sum(prefix_mask[0]))

    print(f"Generating (max {max_new_tokens} tokens)...")
    for step in range(max_new_tokens):
        # embed 当前 token
        token_emb = model.PaliGemma.llm(next_token, method="embed")  # (1, 1, D)

        # position
        pos = jnp.array([[prefix_len + step]])

        # attention mask: 当前 token attend 到所有 prefix
        cur_mask = jnp.ones((1, 1), dtype=bool)
        full_mask = jnp.concatenate([prefix_mask, cur_mask], axis=1)  # (1, prefix+1)
        attn_mask = full_mask[:, None, :]  # (1, 1, prefix+1)

        # forward
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, token_emb],
            mask=attn_mask,
            positions=pos,
            kv_cache=kv_cache
        )

        # logits via tied embedding
        emb_table = model.PaliGemma.llm.embedder.input_embedding.value  # (vocab, D)
        logits = suffix_out[0, 0] @ emb_table.T  # (vocab,)
        next_token_id = int(jnp.argmax(logits))

        if next_token_id == EOS_TOKEN:
            print(f"EOS at step {step}")
            break

        generated_tokens.append(next_token_id)
        next_token = jnp.array([[next_token_id]], dtype=jnp.int32)

        if step % 10 == 0:
            partial = tokenizer.Decode(generated_tokens)
            print(f"  step {step}: ...{partial}")

    return generated_tokens


def main():
    print(f"JAX devices: {jax.devices()}")

    # 1. 加载模型
    model = load_model(NPZ_PATH)
    print(f"Model loaded: {type(model)}")

    # 2. 加载 tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    print(f"Vocab size: {tokenizer.GetPieceSize()}")

    # 3. 处理图像
    print(f"Loading image from {IMAGE_PATH}...")
    image = preprocess_image(IMAGE_PATH)
    print(f"Image shape: {image.shape}")

    # 4. tokenize prompt
    prompt_tokens = tokenizer.Encode(PROMPT)
    print(f"Prompt: '{PROMPT}'")
    print(f"Prompt tokens: {prompt_tokens}")

    # 5. 构造 Observation
    obs = build_observation(image, prompt_tokens, max_token_len=180)

    # 6. 生成
    generated_tokens = generate(model, obs, tokenizer, MAX_NEW_TOKENS)
    result = tokenizer.Decode(generated_tokens)

    print("\n" + "=" * 50)
    print(f"Prompt:  {PROMPT}")
    print(f"Output:  {result}")
    print("=" * 50)


if __name__ == "__main__":
    main()