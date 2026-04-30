#!/usr/bin/env python3
"""Convert ms-swift joint PaliGemma PyTorch checkpoint (.pt) to official-style JAX .npz.

The swift checkpoint stores:
  - Vision + projector + embeddings at top-level ``model.*``
  - Gemma decoder weights under ``model.joint_model.mixtures.vlm.*``
  - Action / proprio / other mixture weights under ``joint_model`` (ignored here)

Output keys match HuggingFace ``convert_paligemma_weights_to_hf.py`` conventions:
flat dict with ``params/img/...`` and ``params/llm/...`` so OpenPI
``PaliGemmaWeightLoader`` can load them (``unflatten_dict(..., sep='/')['params']``).

Large checkpoints: ``torch.load(..., mmap=True)``. Large tensors avoid an intermediate
float32 copy when ``--dtype float16`` (uses ``tensor.half().numpy()``). The embedding
table is converted in row chunks. Arrays are written to the ``.npz`` zip one-by-one so
peak RAM stays near one large buffer plus mmap.

Example:
  PYTHONPATH=src python scripts/swift_pt_paligemma_to_jax_npz.py \\
    --pt /path/to/cot_PaliGemma_ck_15000.pt \\
    --out /path/to/paligemma-3b-pt-224.npz \\
    --truncate_vocab 257152 \\
    --dtype float16

Then verify in-repo loading::

  PYTHONPATH=src python scripts/smoke_test_swift_npz_weights.py --npz /path/to/paligemma-3b-pt-224.npz
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.lib import format as npyformat


def _to_numpy(t, dtype: np.dtype) -> np.ndarray:
    """Materialize a *single* torch tensor; prefer half path to skip float32 peaks."""
    import torch

    if dtype == np.float16:
        return t.detach().cpu().half().numpy()
    return t.detach().cpu().float().numpy().astype(np.float32)


def _build_hf_style_state(model: dict[str, Any]) -> dict[str, Any]:
    """Map swift ``model.*`` keys to the names expected by HF PaliGemma conversion."""
    out: dict[str, Any] = {}
    for k, v in model.items():
        if k.startswith("model.vision_tower."):
            out[k[len("model.") :]] = v
        elif k == "model.multi_modal_projector.linear.weight":
            out["multi_modal_projector.linear.weight"] = v
        elif k == "model.multi_modal_projector.linear.bias":
            out["multi_modal_projector.linear.bias"] = v
        elif k == "model.embed_tokens.weight":
            out["language_model.model.embed_tokens.weight"] = v
        elif k == "model.lm_head.weight":
            out["language_model.lm_head.weight"] = v
        elif k.startswith("model.joint_model.mixtures.vlm.layers."):
            out[k.replace("model.joint_model.mixtures.vlm.", "language_model.model.")] = v
        elif k == "model.joint_model.mixtures.vlm.norm.weight":
            out["language_model.model.norm.weight"] = v
    return out


def _write_npz_like_numpy(out_path: str, arrays: Iterator[tuple[str, np.ndarray]]) -> int:
    """Write the same zip layout as ``numpy.savez`` (one ``key.npy`` per array). Returns count.

    Arrays are serialized to a temporary ``.npy`` on disk then ``ZipFile.write``'d so we do not
    hold a second full copy in a ``BytesIO`` (critical for multi-gigabyte tensors).
    """
    n = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays:
            fd, tmp = tempfile.mkstemp(suffix=".npy")
            os.close(fd)
            try:
                with open(tmp, "wb") as f:
                    npyformat.write_array(f, arr, allow_pickle=False)
                zf.write(tmp, arcname=name + ".npy")
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            del arr
            n += 1
            if n % 2 == 0:
                gc.collect()
    gc.collect()
    return n


def _iter_prefixed_jax_arrays(
    sd: dict[str, Any],
    *,
    scratch_paths: list[str],
    dtype: np.dtype,
    embed_chunk: int = 4096,
    text_num_layers: int = 18,
    text_num_heads: int = 8,
    text_num_kv_heads: int = 1,
    text_head_dim: int = 256,
    text_hidden: int = 2048,
    text_intermediate: int = 16384,
    vision_num_layers: int = 27,
    vision_hidden: int = 1152,
    vision_num_heads: int = 16,
    vision_head_dim: int = 72,
    vision_mlp_dim: int = 4304,
    num_image_tokens: int = 256,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(params/... , ndarray)`` pairs; pops ``sd`` keys when done to drop mmap refs."""
    import torch

    def popg(name: str) -> torch.Tensor:
        if name not in sd:
            raise KeyError(f"Missing HF-style key {name!r} in extracted state dict.")
        return sd.pop(name)

    # --- Vision ---
    w = popg("vision_tower.vision_model.embeddings.patch_embedding.weight")
    yield "params/img/embedding/kernel", _to_numpy(w.permute(2, 3, 1, 0).contiguous(), dtype)
    del w

    yield "params/img/embedding/bias", _to_numpy(
        popg("vision_tower.vision_model.embeddings.patch_embedding.bias"), dtype
    )

    pos = popg("vision_tower.vision_model.embeddings.position_embedding.weight")
    yield "params/img/pos_embedding", _to_numpy(pos.reshape(1, num_image_tokens, vision_hidden), dtype)
    del pos

    ln0_s = np.zeros((vision_num_layers, vision_hidden), dtype=dtype)
    ln0_b = np.zeros((vision_num_layers, vision_hidden), dtype=dtype)
    ln1_s = np.zeros((vision_num_layers, vision_hidden), dtype=dtype)
    ln1_b = np.zeros((vision_num_layers, vision_hidden), dtype=dtype)
    d0_k = np.zeros((vision_num_layers, vision_hidden, vision_mlp_dim), dtype=dtype)
    d0_b = np.zeros((vision_num_layers, vision_mlp_dim), dtype=dtype)
    d1_k = np.zeros((vision_num_layers, vision_mlp_dim, vision_hidden), dtype=dtype)
    d1_b = np.zeros((vision_num_layers, vision_hidden), dtype=dtype)
    # Flax nn.MultiHeadDotProductAttention: q/k/v DenseGeneral kernels are (embed, num_heads, head_dim);
    # out merges (num_heads, head_dim) -> embed so out kernel is (num_heads, head_dim, embed).
    ak = np.zeros((vision_num_layers, vision_hidden, vision_num_heads, vision_head_dim), dtype=dtype)
    ab = np.zeros((vision_num_layers, vision_num_heads, vision_head_dim), dtype=dtype)
    av = np.zeros_like(ak)
    abv = np.zeros_like(ab)
    aq = np.zeros_like(ak)
    abq = np.zeros_like(ab)
    ao = np.zeros((vision_num_layers, vision_num_heads, vision_head_dim, vision_hidden), dtype=dtype)
    abo = np.zeros((vision_num_layers, vision_hidden), dtype=dtype)

    def inv_attn_kernel_qkv(hf_w: torch.Tensor) -> np.ndarray:
        """HF linear (out,in) -> Flax (in, heads, head_dim)."""
        t = hf_w.T.reshape(vision_num_heads, vision_head_dim, vision_hidden)
        return np.transpose(_to_numpy(t, dtype), (2, 0, 1))

    def inv_attn_kernel_out(hf_w: torch.Tensor) -> np.ndarray:
        t = hf_w.T.reshape(vision_num_heads, vision_head_dim, vision_hidden)
        return _to_numpy(t, dtype)

    def inv_attn_bias(hf_b: torch.Tensor) -> np.ndarray:
        t = hf_b.reshape(vision_num_heads, vision_head_dim)
        return _to_numpy(t, dtype)

    for i in range(vision_num_layers):
        p = f"vision_tower.vision_model.encoder.layers.{i}."
        ln0_s[i] = _to_numpy(popg(p + "layer_norm1.weight"), dtype)
        ln0_b[i] = _to_numpy(popg(p + "layer_norm1.bias"), dtype)
        ln1_s[i] = _to_numpy(popg(p + "layer_norm2.weight"), dtype)
        ln1_b[i] = _to_numpy(popg(p + "layer_norm2.bias"), dtype)
        d0_k[i] = _to_numpy(popg(p + "mlp.fc1.weight").T.contiguous(), dtype)
        d0_b[i] = _to_numpy(popg(p + "mlp.fc1.bias"), dtype)
        d1_k[i] = _to_numpy(popg(p + "mlp.fc2.weight").T.contiguous(), dtype)
        d1_b[i] = _to_numpy(popg(p + "mlp.fc2.bias"), dtype)
        ak[i] = inv_attn_kernel_qkv(popg(p + "self_attn.k_proj.weight"))
        ab[i] = inv_attn_bias(popg(p + "self_attn.k_proj.bias"))
        av[i] = inv_attn_kernel_qkv(popg(p + "self_attn.v_proj.weight"))
        abv[i] = inv_attn_bias(popg(p + "self_attn.v_proj.bias"))
        aq[i] = inv_attn_kernel_qkv(popg(p + "self_attn.q_proj.weight"))
        abq[i] = inv_attn_bias(popg(p + "self_attn.q_proj.bias"))
        ao[i] = inv_attn_kernel_out(popg(p + "self_attn.out_proj.weight"))
        abo[i] = _to_numpy(popg(p + "self_attn.out_proj.bias"), dtype)

    yield "params/img/Transformer/encoderblock/LayerNorm_0/scale", ln0_s
    yield "params/img/Transformer/encoderblock/LayerNorm_0/bias", ln0_b
    yield "params/img/Transformer/encoderblock/LayerNorm_1/scale", ln1_s
    yield "params/img/Transformer/encoderblock/LayerNorm_1/bias", ln1_b
    yield "params/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel", d0_k
    yield "params/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias", d0_b
    yield "params/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel", d1_k
    yield "params/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias", d1_b
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel", ak
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias", ab
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel", av
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias", abv
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel", aq
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias", abq
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel", ao
    yield "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias", abo

    yield "params/img/Transformer/encoder_norm/scale", _to_numpy(
        popg("vision_tower.vision_model.post_layernorm.weight"), dtype
    )
    yield "params/img/Transformer/encoder_norm/bias", _to_numpy(
        popg("vision_tower.vision_model.post_layernorm.bias"), dtype
    )

    proj_w = popg("multi_modal_projector.linear.weight")
    yield "params/img/head/kernel", _to_numpy(proj_w.T.contiguous(), dtype)
    del proj_w
    yield "params/img/head/bias", _to_numpy(popg("multi_modal_projector.linear.bias"), dtype)

    # --- LLM embedding (chunked into on-disk memmap) ---
    emb_key = "language_model.model.embed_tokens.weight"
    embed = popg(emb_key)
    n_vocab, embed_dim = int(embed.shape[0]), int(embed.shape[1])
    e_fd, e_path = tempfile.mkstemp(prefix="paligemma_embed_", suffix=".dat")
    os.close(e_fd)
    scratch_paths.append(e_path)
    emb_mm = np.memmap(e_path, dtype=dtype, mode="w+", shape=(n_vocab, embed_dim))
    for start in range(0, n_vocab, embed_chunk):
        end = min(start + embed_chunk, n_vocab)
        sl = embed[start:end]
        if dtype == np.float16:
            emb_mm[start:end] = sl.detach().cpu().half().numpy()
        else:
            emb_mm[start:end] = sl.detach().cpu().float().numpy().astype(np.float32)
        del sl
    del embed
    emb_mm.flush()
    gc.collect()
    yield "params/llm/embedder/input_embedding", emb_mm
    del emb_mm
    gc.collect()

    # Drop tied lm_head if still present (not part of JAX npz).
    sd.pop("language_model.lm_head.weight", None)

    q_w = np.zeros((text_num_layers, text_num_heads, text_hidden, text_head_dim), dtype=dtype)
    # OpenPI Gemma GQA: kv_einsum is (2, num_kv_heads, width, head_dim); PaliGemma 2B uses num_kv_heads=1.
    kv_w = np.zeros((text_num_layers, 2, text_num_kv_heads, text_hidden, text_head_dim), dtype=dtype)
    o_w = np.zeros((text_num_layers, text_num_heads, text_head_dim, text_hidden), dtype=dtype)

    g_fd, g_path = tempfile.mkstemp(prefix="paligemma_gating_", suffix=".dat")
    os.close(g_fd)
    scratch_paths.append(g_path)
    gating = np.memmap(g_path, dtype=dtype, mode="w+", shape=(text_num_layers, 2, text_hidden, text_intermediate))

    l_fd, l_path = tempfile.mkstemp(prefix="paligemma_linear_", suffix=".dat")
    os.close(l_fd)
    scratch_paths.append(l_path)
    linear = np.memmap(l_path, dtype=dtype, mode="w+", shape=(text_num_layers, text_intermediate, text_hidden))

    pre_attn = np.zeros((text_num_layers, text_hidden), dtype=dtype)
    pre_ff = np.zeros((text_num_layers, text_hidden), dtype=dtype)

    for i in range(text_num_layers):
        p = f"language_model.model.layers.{i}."
        hf_q = popg(p + "self_attn.q_proj.weight")
        y = hf_q.reshape(text_num_heads, text_head_dim, text_hidden).permute(0, 2, 1).contiguous()
        q_w[i] = _to_numpy(y, dtype)
        del hf_q, y

        hf_k = popg(p + "self_attn.k_proj.weight")
        hf_v = popg(p + "self_attn.v_proj.weight")
        kv_w[i, 0, 0] = _to_numpy(hf_k.T.contiguous(), dtype)
        kv_w[i, 1, 0] = _to_numpy(hf_v.T.contiguous(), dtype)
        del hf_k, hf_v

        hf_o = popg(p + "self_attn.o_proj.weight")
        z = hf_o.reshape(text_hidden, text_num_heads, text_head_dim).permute(1, 2, 0).contiguous()
        o_w[i] = _to_numpy(z, dtype)
        del hf_o, z

        hf_gate = popg(p + "mlp.gate_proj.weight")
        hf_up = popg(p + "mlp.up_proj.weight")
        gating[i, 0] = _to_numpy(hf_gate.T.contiguous(), dtype)
        gating[i, 1] = _to_numpy(hf_up.T.contiguous(), dtype)
        del hf_gate, hf_up

        hf_down = popg(p + "mlp.down_proj.weight")
        linear[i] = _to_numpy(hf_down.T.contiguous(), dtype)
        del hf_down

        pre_attn[i] = _to_numpy(popg(p + "input_layernorm.weight"), dtype)
        pre_ff[i] = _to_numpy(popg(p + "post_attention_layernorm.weight"), dtype)

    yield "params/llm/layers/attn/q_einsum/w", q_w
    yield "params/llm/layers/attn/kv_einsum/w", kv_w
    yield "params/llm/layers/attn/attn_vec_einsum/w", o_w
    gating.flush()
    yield "params/llm/layers/mlp/gating_einsum", gating
    del gating
    gc.collect()
    linear.flush()
    yield "params/llm/layers/mlp/linear", linear
    del linear
    gc.collect()
    yield "params/llm/layers/pre_attention_norm/scale", pre_attn
    yield "params/llm/layers/pre_ffw_norm/scale", pre_ff
    yield "params/llm/final_norm/scale", _to_numpy(popg("language_model.model.norm.weight"), dtype)

    if sd:
        keys = ", ".join(sorted(sd.keys())[:20])
        raise RuntimeError(f"Unused keys remain in extracted state dict (first 20): {keys}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt", required=True, help="Path to ms-swift .pt checkpoint (mmap load).")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument(
        "--truncate_vocab",
        type=int,
        default=257152,
        help="If >0, slice ``llm/embedder/input_embedding`` to this many rows (OpenPI default 257152). "
        "Use 0 to keep full embedding from the checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Numpy dtype for stored arrays (bfloat16 not supported in npz).",
    )
    parser.add_argument(
        "--embed_chunk",
        type=int,
        default=4096,
        help="Row chunk size when converting the token embedding matrix.",
    )
    parser.add_argument(
        "--reference_npz",
        default="",
        help="Optional official paligemma .npz: assert key set matches.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as e:
        print("This script requires PyTorch: pip install torch", file=sys.stderr)
        raise SystemExit(1) from e

    dtype = np.float16 if args.dtype == "float16" else np.float32

    print(f"Loading (mmap): {args.pt}", flush=True)
    ckpt = torch.load(args.pt, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise SystemExit("Expected a dict checkpoint with a 'model' key.")
    raw_model = ckpt["model"]

    print("Extracting PaliGemma (vision + projector + vlm mixture) …", flush=True)
    hf_sd = _build_hf_style_state(raw_model)
    del ckpt, raw_model
    gc.collect()

    emb_key = "language_model.model.embed_tokens.weight"
    n_vocab = int(hf_sd[emb_key].shape[0])
    if args.truncate_vocab and args.truncate_vocab > 0:
        if n_vocab < args.truncate_vocab:
            raise SystemExit(f"Vocab {n_vocab} < truncate_vocab {args.truncate_vocab}")
        if n_vocab != args.truncate_vocab:
            print(
                f"Truncating embedding from {n_vocab} -> {args.truncate_vocab} rows "
                f"(tail tokens dropped; matches OpenPI PALIGEMMA_VOCAB_SIZE).",
                flush=True,
            )
            hf_sd[emb_key] = hf_sd[emb_key][: args.truncate_vocab]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    print(f"Writing to {args.out} …", flush=True)

    scratch_mm_files: list[str] = []

    def arrays_iter() -> Iterator[tuple[str, np.ndarray]]:
        yield from _iter_prefixed_jax_arrays(
            hf_sd,
            scratch_paths=scratch_mm_files,
            dtype=dtype,
            embed_chunk=args.embed_chunk,
        )

    n = _write_npz_like_numpy(args.out, arrays_iter())
    for p in scratch_mm_files:
        try:
            os.unlink(p)
        except OSError:
            pass

    if args.reference_npz:
        ref = np.load(args.reference_npz, allow_pickle=False)
        out = np.load(args.out, allow_pickle=False)
        ref_keys, out_keys = set(ref.files), set(out.files)
        if ref_keys != out_keys:
            print("WARNING: key mismatch vs reference.", flush=True)
            print("  only in ref:", sorted(ref_keys - out_keys)[:30], flush=True)
            print("  only in out:", sorted(out_keys - ref_keys)[:30], flush=True)
        for k in sorted(ref_keys & out_keys):
            if ref[k].shape != out[k].shape:
                raise SystemExit(f"Shape mismatch for {k}: ref {ref[k].shape} vs out {out[k].shape}")

    del hf_sd
    gc.collect()
    print(f"Done. Wrote {n} arrays.", flush=True)
    _sanity_check_output_npz(args.out)


def _sanity_check_output_npz(path: str) -> None:
    """Assert critical tensors match OpenPI PaliGemma / Gemma 2B layout (fail fast before training)."""
    # NpzFile ``.files`` lists array names *without* the ``.npy`` suffix (zip members still end in ``.npy``).
    kv_key = "params/llm/layers/attn/kv_einsum/w"
    img_key_k = "params/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"
    want_kv = (18, 2, 1, 2048, 256)
    want_img_k = (27, 1152, 16, 72)
    with np.load(path, allow_pickle=False) as z:
        if kv_key not in z.files:
            raise SystemExit(f"Sanity check: missing {kv_key!r} in {path} (files sample: {list(z.files)[:8]} ...)")
        sh = tuple(z[kv_key].shape)
        if sh != want_kv:
            raise SystemExit(f"Sanity check: {kv_key} has shape {sh}, expected {want_kv}")
        if img_key_k in z.files:
            sh_i = tuple(z[img_key_k].shape)
            if sh_i != want_img_k:
                raise SystemExit(f"Sanity check: {img_key_k} has shape {sh_i}, expected {want_img_k}")
    print(f"Sanity check OK: {kv_key} {want_kv}, {img_key_k} {want_img_k}", flush=True)


if __name__ == "__main__":
    main()
