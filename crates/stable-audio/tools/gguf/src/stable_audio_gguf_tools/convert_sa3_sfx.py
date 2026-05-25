from __future__ import annotations

import argparse
import json
import logging
import hashlib
from pathlib import Path
from typing import Iterable

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm

LOG = logging.getLogger("stable-audio-gguf")
DEFAULT_REPO_ID = "stabilityai/stable-audio-3-small-sfx"
GGML_MAX_NAME = 63


NAME_REPLACEMENTS = (
    ("conditioner.conditioners.", "cond."),
    ("seconds_total.embedder.embedding.", "sec."),
    ("padding_embedding", "pad_emb"),
    ("transformer.layers.", "tr.blk."),
    ("transformers.", "tr."),
    ("self_attn.", "sa."),
    ("cross_attn.", "ca."),
    ("cross_attend_norm.", "ca_norm."),
    ("global_cond_embedder.", "gce."),
    ("to_scale_shift_gate", "ssg"),
    ("to_local_embed.", "loc."),
    ("to_cond_embed.", "cond_emb."),
    ("to_global_embed.", "glob_emb."),
    ("to_timestep_embed.", "time_emb."),
    ("preprocess_conv.", "pre."),
    ("postprocess_conv.", "post."),
    ("project_in.", "pin."),
    ("project_out.", "pout."),
    ("pre_norm.", "pn."),
    ("ff_norm.", "fn."),
    ("q_norm.", "qn."),
    ("k_norm.", "kn."),
    ("to_qkv.", "qkv."),
    ("to_kv.", "kv."),
    ("to_q.", "q."),
    ("to_out.", "out."),
    ("ff.ff.0.proj.", "ff0."),
    ("ff.ff.2.", "ff2."),
    ("layers.", "ly."),
    ("embed_tokens.", "emb."),
    ("pre_self_attn_layernorm.", "pre_sa_ln."),
    ("post_self_attn_layernorm.", "post_sa_ln."),
    ("pre_feedforward_layernorm.", "pre_ff_ln."),
    ("post_feedforward_layernorm.", "post_ff_ln."),
    ("self_attn.", "sa."),
    ("gate_proj.", "gate."),
    ("up_proj.", "up."),
    ("down_proj.", "down."),
    ("q_proj.", "q."),
    ("k_proj.", "k."),
    ("v_proj.", "v."),
    ("o_proj.", "o."),
    ("bottleneck.", "bn."),
    ("decoder.layers.", "dec.ly."),
    ("mapping.", "map."),
)


def _short_name(name: str) -> str:
    out = name
    for old, new in NAME_REPLACEMENTS:
        out = out.replace(old, new)
    if len(out) <= GGML_MAX_NAME:
        return out
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    keep = GGML_MAX_NAME - len(digest) - 1
    return f"{out[:keep]}.{digest}"


def _should_quantize(tensor_name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim <= 1:
        return False
    if tensor_name.endswith(".bias"):
        return False
    if any(
        part in tensor_name
        for part in (
            ".gamma",
            ".beta",
            ".alpha",
            "pad_emb",
            "running_std",
            "new_tokens",
            "cond.sec.",
        )
    ):
        return False
    return True


def _tensor_to_np(
    tensor_name: str,
    tensor: torch.Tensor,
    out_type: str,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    tensor = tensor.detach().cpu()
    if out_type == "f32" or tensor.ndim <= 1:
        return tensor.float().numpy().astype(np.float32), gguf.GGMLQuantizationType.F32
    data = tensor.float().numpy()
    if out_type == "f16" or not _should_quantize(tensor_name, tensor):
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16
    quant_type = {
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q6_k": gguf.GGMLQuantizationType.Q6_K,
    }[out_type]
    try:
        return gguf.quants.quantize(data.astype(np.float32), quant_type), quant_type
    except Exception as exc:
        LOG.warning("%s quantization failed for %s: %s; falling back to F16", out_type, tensor_name, exc)
        return data.astype(np.float16), gguf.GGMLQuantizationType.F16


def _write_gguf(
    output: Path,
    arch: str,
    name: str,
    metadata: dict[str, object],
    tensors: Iterable[tuple[str, torch.Tensor]],
    out_type: str,
    tokenizer_json: str | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path=None, arch=arch)
    writer.add_name(name)
    writer.add_type(gguf.GGUFType.MODEL)
    file_type = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
    }[out_type]
    writer.add_file_type(file_type)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    for key, value in metadata.items():
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            writer.add_string(key, json.dumps(value, sort_keys=True))

    if tokenizer_json is not None:
        writer.add_string("tokenizer.huggingface.json", tokenizer_json)

    count = 0
    tensor_items = list(tensors)
    used_names: dict[str, str] = {}
    for tensor_name, tensor in tqdm(tensor_items, desc=f"writing {output.name}"):
        if tensor.numel() == 0:
            LOG.info("skipping zero-sized tensor: %s", tensor_name)
            continue
        tensor_name = _short_name(tensor_name)
        if tensor_name in used_names:
            raise RuntimeError(f"short tensor name collision: {tensor_name}")
        used_names[tensor_name] = tensor_name
        data, dtype = _tensor_to_np(tensor_name, tensor, out_type)
        writer.add_tensor(tensor_name, data, raw_dtype=dtype)
        count += 1

    writer.write_header_to_file(path=output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    return count


def _iter_safetensors(path: Path, prefixes: tuple[str, ...], strip_prefix: str = ""):
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if prefixes and not key.startswith(prefixes):
                continue
            out = key[len(strip_prefix) :] if strip_prefix and key.startswith(strip_prefix) else key
            yield out, f.get_tensor(key)


def _iter_t5_encoder(path: Path):
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith("model.encoder."):
                continue
            out = key[len("model.encoder.") :]
            tensor = f.get_tensor(key)
            # Gemma RMSNorm uses x * (1 + weight). Store the effective scale so
            # the Rust ggml graph can use regular RMSNorm * weight.
            if out.endswith("layernorm.weight") or out == "norm.weight":
                tensor = tensor.float() + 1.0
            yield out, tensor


def _iter_same_decoder(path: Path):
    prefix = "pretransform.model."
    weight_g_key = prefix + "decoder.layers.3.mapping.weight_g"
    weight_v_key = prefix + "decoder.layers.3.mapping.weight_v"
    weight_key = prefix + "decoder.layers.3.mapping.weight"
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        fused_mapping = None
        if weight_g_key in keys and weight_v_key in keys:
            weight_g = f.get_tensor(weight_g_key).float()
            weight_v = f.get_tensor(weight_v_key).float()
            fused_mapping = weight_v * (
                weight_g / weight_v.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            )
        for key in f.keys():
            if not key.startswith(("pretransform.model.decoder.", "pretransform.model.bottleneck.")):
                continue
            if key in (weight_g_key, weight_v_key):
                continue
            if fused_mapping is not None and key == weight_key:
                continue
            out = key[len(prefix) :] if key.startswith(prefix) else key
            yield out, f.get_tensor(key)
        if fused_mapping is not None:
            yield "decoder.layers.3.mapping.weight", fused_mapping


def _iter_same_encoder(path: Path):
    prefix = "pretransform.model."
    weight_g_key = prefix + "encoder.layers.0.mapping.weight_g"
    weight_v_key = prefix + "encoder.layers.0.mapping.weight_v"
    weight_key = prefix + "encoder.layers.0.mapping.weight"
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        fused_mapping = None
        if weight_g_key in keys and weight_v_key in keys:
            weight_g = f.get_tensor(weight_g_key).float()
            weight_v = f.get_tensor(weight_v_key).float()
            fused_mapping = weight_v * (
                weight_g / weight_v.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
            )
        for key in f.keys():
            if not key.startswith(("pretransform.model.encoder.", "pretransform.model.bottleneck.")):
                continue
            if key in (weight_g_key, weight_v_key):
                continue
            if fused_mapping is not None and key == weight_key:
                continue
            out = key[len(prefix) :] if key.startswith(prefix) else key
            yield out, f.get_tensor(key)
        if fused_mapping is not None:
            yield "encoder.layers.0.mapping.weight", fused_mapping


def _repo_snapshot(local_dir: Path | None, repo_id: str) -> Path:
    if local_dir is not None:
        return local_dir
    return Path(
        snapshot_download(
            repo_id,
            allow_patterns=[
                "model.safetensors",
                "model_config.json",
                "t5gemma-b-b-ul2/model.safetensors",
                "t5gemma-b-b-ul2/config.json",
                "t5gemma-b-b-ul2/tokenizer.json",
                "t5gemma-b-b-ul2/tokenizer_config.json",
            ],
        )
    )


def _model_slug(repo_id: str) -> str:
    name = repo_id.rsplit("/", 1)[-1]
    return name.removeprefix("stable-audio-3-")


def convert(snapshot: Path, output_dir: Path, out_type: str, repo_id: str) -> None:
    model_config_path = snapshot / "model_config.json"
    t5_config_path = snapshot / "t5gemma-b-b-ul2" / "config.json"
    tokenizer_path = snapshot / "t5gemma-b-b-ul2" / "tokenizer.json"
    model_path = snapshot / "model.safetensors"
    t5_path = snapshot / "t5gemma-b-b-ul2" / "model.safetensors"

    model_config = json.loads(model_config_path.read_text())
    t5_config = json.loads(t5_config_path.read_text())
    tokenizer_json = tokenizer_path.read_text()
    dit_cfg = model_config["model"]["diffusion"]["config"]
    ae_cfg = model_config["model"]["pretransform"]["config"]
    t5_encoder_cfg = t5_config["encoder"]

    common = {
        "stable_audio.arch": "stable-audio-3",
        "stable_audio.model_id": repo_id,
        "stable_audio.sample_rate": int(model_config["sample_rate"]),
        "stable_audio.audio_channels": int(model_config["audio_channels"]),
    }
    slug = _model_slug(repo_id)
    is_legacy_sfx = slug == "small-sfx"
    decoder_kind = "same-l" if slug == "medium" else "same-s"
    dit_filename = "sa3-small-sfx-dit.gguf" if is_legacy_sfx else f"sa3-{slug}-dit.gguf"
    decoder_filename = (
        "sa3-same-s-decoder.gguf" if is_legacy_sfx else f"sa3-{slug}-{decoder_kind}-decoder.gguf"
    )
    encoder_filename = (
        "sa3-same-s-encoder.gguf" if is_legacy_sfx else f"sa3-{slug}-{decoder_kind}-encoder.gguf"
    )
    title = slug.replace("-", " ").title()

    dit_meta = {
        **common,
        "sa3.component": f"dit-{slug}",
        "sa3.io_channels": int(dit_cfg["io_channels"]),
        "sa3.embed_dim": int(dit_cfg["embed_dim"]),
        "sa3.depth": int(dit_cfg["depth"]),
        "sa3.num_heads": int(dit_cfg["num_heads"]),
        "sa3.cond_token_dim": int(dit_cfg["cond_token_dim"]),
        "sa3.global_cond_dim": int(dit_cfg["global_cond_dim"]),
        "sa3.local_add_cond_dim": int(dit_cfg["local_add_cond_dim"]),
        "sa3.num_memory_tokens": int(dit_cfg["num_memory_tokens"]),
    }
    n = _write_gguf(
        output_dir / dit_filename,
        f"sa3-{slug}-dit",
        f"Stable Audio 3 {title} DiT",
        dit_meta,
        _iter_safetensors(model_path, ("model.model.", "conditioner."), strip_prefix="model.model."),
        out_type,
    )
    LOG.info("wrote DiT tensors: %s", n)

    dec_meta = {
        **common,
        "sa3.component": f"{decoder_kind}-decoder",
        "sa3.latent_dim": int(ae_cfg["latent_dim"]),
        "sa3.downsampling_ratio": int(ae_cfg["downsampling_ratio"]),
        "sa3.patch_size": int(ae_cfg["pretransform"]["config"]["patch_size"]),
        "sa3.decoder_channels": int(ae_cfg["decoder"]["config"]["channels"]),
    }
    n = _write_gguf(
        output_dir / decoder_filename,
        f"sa3-{decoder_kind}-decoder",
        f"{title} {decoder_kind.upper()} decoder",
        dec_meta,
        _iter_same_decoder(model_path),
        out_type,
    )
    LOG.info("wrote decoder tensors: %s", n)

    enc_meta = {
        **common,
        "sa3.component": f"{decoder_kind}-encoder",
        "sa3.latent_dim": int(ae_cfg["latent_dim"]),
        "sa3.downsampling_ratio": int(ae_cfg["downsampling_ratio"]),
        "sa3.patch_size": int(ae_cfg["pretransform"]["config"]["patch_size"]),
        "sa3.encoder_channels": int(ae_cfg["encoder"]["config"]["channels"]),
    }
    n = _write_gguf(
        output_dir / encoder_filename,
        f"sa3-{decoder_kind}-encoder",
        f"{title} {decoder_kind.upper()} encoder",
        enc_meta,
        _iter_same_encoder(model_path),
        out_type,
    )
    LOG.info("wrote encoder tensors: %s", n)

    t5_meta = {
        **common,
        "sa3.component": "t5gemma-encoder",
        "t5gemma.hidden_size": int(t5_encoder_cfg["hidden_size"]),
        "t5gemma.num_hidden_layers": int(t5_encoder_cfg["num_hidden_layers"]),
        "t5gemma.num_attention_heads": int(t5_encoder_cfg["num_attention_heads"]),
        "t5gemma.num_key_value_heads": int(t5_encoder_cfg["num_key_value_heads"]),
        "t5gemma.head_dim": int(t5_encoder_cfg["head_dim"]),
        "t5gemma.intermediate_size": int(t5_encoder_cfg["intermediate_size"]),
        "t5gemma.vocab_size": int(t5_encoder_cfg["vocab_size"]),
        "t5gemma.rope_theta": float(t5_encoder_cfg["rope_theta"]),
        "t5gemma.rms_norm_eps": float(t5_encoder_cfg["rms_norm_eps"]),
        "t5gemma.pad_token_id": int(t5_config["pad_token_id"]),
    }
    n = _write_gguf(
        output_dir / "t5gemma-b-b-ul2-encoder.gguf",
        "t5gemma-encoder",
        "T5Gemma B/B UL2 encoder",
        t5_meta,
        _iter_t5_encoder(t5_path),
        out_type,
        tokenizer_json=tokenizer_json,
    )
    LOG.info("wrote T5Gemma tensors: %s", n)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Convert Stable Audio 3 safetensors to GGUF")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_REPO_ID,
        choices=[
            "stabilityai/stable-audio-3-small-sfx",
            "stabilityai/stable-audio-3-small-music",
            "stabilityai/stable-audio-3-medium",
        ],
        help="Hugging Face model id to convert",
    )
    parser.add_argument("--snapshot", type=Path, default=None, help="Local HF snapshot directory")
    parser.add_argument("--output-dir", type=Path, default=Path("models/gguf"))
    parser.add_argument("--out-type", choices=["f16", "f32", "q8_0", "q6_k"], default="f16")
    args = parser.parse_args()

    snapshot = _repo_snapshot(args.snapshot, args.model_id)
    LOG.info("using snapshot: %s", snapshot)
    convert(snapshot, args.output_dir, args.out_type, args.model_id)
