# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# load_model_checkpoint.py — RealRebelAI/ComfyUI_MagiHuman_fp8_ditFIX_nodes
# Zero-mmap monolithic FP8 loader for 16GB RAM / RTX 3070 environments
# Requires: PyTorch >= 2.1 (for float8_e4m3fn), safetensors (any version)

import struct
import json
import gc
import logging
from typing import Dict, Iterator, Tuple, Optional

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# LAYER 1 — dtype maps (safetensors → torch)
# ──────────────────────────────────────────────

try:
    _fp8_e4m3 = torch.float8_e4m3fn
    _fp8_e5m2 = torch.float8_e5m2
    _FP8_SUPPORTED = True
except AttributeError:
    _fp8_e4m3 = None
    _fp8_e5m2 = None
    _FP8_SUPPORTED = False

_ST_DTYPE = {
    "F64":  torch.float64,
    "F32":  torch.float32,
    "F16":  torch.float16,
    "BF16": torch.bfloat16,
    "I64":  torch.int64,
    "I32":  torch.int32,
    "I16":  torch.int16,
    "I8":   torch.int8,
    "U8":   torch.uint8,
    "BOOL": torch.bool,
}
if _FP8_SUPPORTED:
    _ST_DTYPE["F8_E4M3"] = _fp8_e4m3
    _ST_DTYPE["F8_E5M2"] = _fp8_e5m2

_FROMBUFFER_AS_UINT8 = {torch.bool}
if _FP8_SUPPORTED:
    _FROMBUFFER_AS_UINT8.add(_fp8_e4m3)
    _FROMBUFFER_AS_UINT8.add(_fp8_e5m2)


# ──────────────────────────────────────────────
# LAYER 2 — Zero-mmap header parser
# ──────────────────────────────────────────────

def _parse_header(path: str) -> Tuple[Dict, int]:
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"File too small to be a valid safetensors: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    data_start = 8 + header_len
    return header, data_start


# ──────────────────────────────────────────────
# LAYER 3 — No-mmap tensor iterator
# ──────────────────────────────────────────────

def _iter_tensors_no_mmap(
    path: str,
    read_chunk: int = 128 * 1024 * 1024,
) -> Iterator[Tuple[str, torch.Tensor]]:
    header, data_start = _parse_header(path)
    tensor_meta = {k: v for k, v in header.items() if k != "__metadata__"}

    with open(path, "rb", buffering=0) as f:
        for key, meta in tensor_meta.items():
            dtype_str  = meta["dtype"]
            shape      = meta["shape"]
            begin, end = meta["data_offsets"]
            abs_begin  = data_start + begin
            n_bytes    = end - begin
            torch_dtype = _ST_DTYPE[dtype_str]

            buf  = bytearray(n_bytes)
            view = memoryview(buf)

            f.seek(abs_begin)
            filled = 0
            while filled < n_bytes:
                want = min(read_chunk, n_bytes - filled)
                got  = f.readinto(view[filled : filled + want])
                if not got:
                    raise EOFError(
                        f"Unexpected EOF reading '{key}' "
                        f"(got {filled}/{n_bytes} bytes)"
                    )
                filled += got

            snap = bytes(buf)
            del buf, view

            raw = torch.frombuffer(snap, dtype=torch.uint8)
            del snap

            if torch_dtype in _FROMBUFFER_AS_UINT8:
                tensor = raw.view(torch_dtype)
            else:
                tensor = torch.frombuffer(
                    raw.numpy().tobytes(), dtype=torch_dtype
                )

            tensor = tensor.reshape(shape).clone()
            del raw

            yield key, tensor


# ──────────────────────────────────────────────
# LAYER 4 — Streaming GPU dispatcher
# ──────────────────────────────────────────────

def load_sharded_safetensors_parallel_with_progress(
    model: torch.nn.Module,
    checkpoint_path: str,
    target_device: Optional[torch.device] = None,
    progress_callback=None,
    sync_every: int = 50,
    gc_every: int = 100,
) -> None:
    if target_device is None:
        import comfy.model_management as mm
        target_device = mm.get_torch_device()

    param_map:  Dict[str, torch.nn.Parameter] = dict(model.named_parameters())
    buffer_map: Dict[str, torch.Tensor]       = dict(model.named_buffers())
    module_map: Dict[str, torch.nn.Module]    = dict(model.named_modules())

    header, _ = _parse_header(checkpoint_path)
    total     = sum(1 for k in header if k != "__metadata__")
    skipped   = []
    loaded    = 0

    logger.info(
        f"[FP8Loader] Streaming '{checkpoint_path}' → {target_device} "
        f"| {total} tensors | no-mmap mode"
    )

    for key, cpu_tensor in _iter_tensors_no_mmap(checkpoint_path):
        gpu_tensor = cpu_tensor.to(device=target_device, non_blocking=True)
        del cpu_tensor

        if key in param_map:
            if param_map[key].shape != gpu_tensor.shape:
                logger.warning(
                    f"[FP8Loader] Shape mismatch '{key}': "
                    f"model={param_map[key].shape} ckpt={gpu_tensor.shape}"
                )
            with torch.no_grad():
                param_map[key].data = gpu_tensor.to(param_map[key].dtype)

        elif key in buffer_map:
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                parent = module_map.get(parts[0])
                if parent is not None:
                    parent.register_buffer(parts[1], gpu_tensor, persistent=True)
                    loaded += 1
                    continue
            buffer_map[key].data = gpu_tensor

        else:
            skipped.append(key)
            loaded += 1
            continue

        loaded += 1

        if progress_callback:
            progress_callback(loaded, total, key)

        if loaded % sync_every == 0:
            torch.cuda.synchronize(target_device)

        if loaded % gc_every == 0:
            gc.collect()
            vram_gb = torch.cuda.memory_allocated(target_device) / 1e9
            logger.info(
                f"[FP8Loader] {loaded}/{total} tensors | "
                f"VRAM: {vram_gb:.2f} GB"
            )

    torch.cuda.synchronize(target_device)
    gc.collect()

    if skipped:
        logger.warning(
            f"[FP8Loader] {len(skipped)} keys not found in model "
            f"(first 10): {skipped[:10]}"
        )
    logger.info(
        f"[FP8Loader] Complete. {loaded - len(skipped)}/{total} tensors loaded."
    )


# ──────────────────────────────────────────────
# PUBLIC ENTRY POINT — called by get_dit.py as:
#   model = load_model_checkpoint(model, engine_config)
# ──────────────────────────────────────────────

def load_model_checkpoint(
    model: torch.nn.Module,
    engine_config,
) -> torch.nn.Module:
    ckpt_path = (
        getattr(engine_config, "load",               None)
        or getattr(engine_config, "dit_checkpoint_path", None)
        or getattr(engine_config, "checkpoint_path",  None)
        or getattr(engine_config, "ckpt_path",        None)
        or getattr(engine_config, "model_path",       None)
    )

    if ckpt_path is None:
        raise ValueError(
            "[FP8Loader] Cannot find checkpoint path in engine_config.\n"
            "Expected one of: dit_checkpoint_path, checkpoint_path, "
            "ckpt_path, model_path.\n"
            f"Got attributes: {list(vars(engine_config).keys())}"
        )

    logger.info(f"[FP8Loader] load_model_checkpoint → '{ckpt_path}'")

    load_sharded_safetensors_parallel_with_progress(
        model=model,
        checkpoint_path=ckpt_path,
    )

    return model
