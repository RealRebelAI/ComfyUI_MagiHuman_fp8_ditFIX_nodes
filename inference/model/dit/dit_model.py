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

import gc

import torch
from ...infra.checkpoint import load_model_checkpoint
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
from ...utils import print_mem_info_rank_0, print_rank_0

from .dit_module import DiTModel

def get_dit(model_config, engine_config, torch_type, offload=False,):
    """Build and load DiT model."""
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        model = DiTModel(model_config=model_config)

    print_rank_0("Build dit model successfully")
    
    # --- FP8 INJECTION FIX: PREPARE MODEL FOR SCALES ---
    # Register the exact key from the error log so load_model_checkpoint catches it
    for module in model.modules():
        if module.__class__.__name__ in ['BaseLinear', 'NativeMoELinear']:
            module.register_buffer('weight.__fp8_scale', torch.ones(1, dtype=torch.float32))

    # Load the checkpoint (this will now map the scales correctly without "Unexpected Keys")
    model = load_model_checkpoint(model, engine_config)

    # --- FP8 INJECTION FIX: SAFE TYPE CASTING ---
    # Replace .to(torch_type) with a selective cast to protect float8 weights and float32 scales
    for param in model.parameters():
        if getattr(param, "dtype", None) != torch.float8_e4m3fn:
            param.data = param.data.to(torch_type)
            
    for name, buf in model.named_buffers():
        if getattr(buf, "dtype", None) != torch.float8_e4m3fn and '__fp8_scale' not in name:
            buf.data = buf.data.to(torch_type)

    if offload:
        model.cuda(torch.cuda.current_device())
    model.eval()
    print_mem_info_rank_0("Load model successfully")

    gc.collect()
    torch.cuda.empty_cache()
    return model
