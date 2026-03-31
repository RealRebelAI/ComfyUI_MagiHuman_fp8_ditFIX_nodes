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
def get_dit(model_config, engine_config,torch_type,offload=False,):
    """Build and load DiT model."""
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with nullcontext():
    model = DiTModel(model_config=model_config)

    print_rank_0("Build dit model successfully")
    #print_rank_0(model)
    '''
        [2026-03-26 16:13:55,782 - INFO] [Rank 0] DiTModel(
    (adapter): Adapter(
        (video_embedder): Linear(in_features=192, out_features=5120, bias=True)
        (text_embedder): Linear(in_features=3584, out_features=5120, bias=True)
        (audio_embedder): Linear(in_features=64, out_features=5120, bias=True)
        (rope): ElementWiseFourierEmbed()
    )
    (block): TransformerBlock(
        (layers): ModuleList(
        (0-3): 4 x TransFormerLayer(
            (attention): Attention(
            (pre_norm): MultiModalityRMSNorm()
            (linear_qkv): NativeMoELinear()
            (linear_proj): NativeMoELinear()
            (q_norm): MultiModalityRMSNorm()
            (k_norm): MultiModalityRMSNorm()
            )
            (mlp): MLP(
            self.up_gate_proj.weight.shape=torch.Size([61440, 5120]), self.down_proj.weight.shape=torch.Size([15360, 20480])
            (pre_norm): MultiModalityRMSNorm()
            (up_gate_proj): NativeMoELinear()
            (down_proj): NativeMoELinear()
            )
        )
        (4-35): 32 x TransFormerLayer(
            (attention): Attention(
            (pre_norm): MultiModalityRMSNorm()
            (linear_qkv): BaseLinear()
            (linear_proj): BaseLinear()
            (q_norm): MultiModalityRMSNorm()
            (k_norm): MultiModalityRMSNorm()
            )
            (mlp): MLP(
            self.up_gate_proj.weight.shape=torch.Size([27304, 5120]), self.down_proj.weight.shape=torch.Size([5120, 13652])
            (pre_norm): MultiModalityRMSNorm()
            (up_gate_proj): BaseLinear()
            (down_proj): BaseLinear()
            )
        )
        (36-39): 4 x TransFormerLayer(
            (attention): Attention(
            (pre_norm): MultiModalityRMSNorm()
            (linear_qkv): NativeMoELinear()
            (linear_proj): NativeMoELinear()
            (q_norm): MultiModalityRMSNorm()
            (k_norm): MultiModalityRMSNorm()
            )
            (mlp): MLP(
            self.up_gate_proj.weight.shape=torch.Size([81912, 5120]), self.down_proj.weight.shape=torch.Size([15360, 13652])
            (pre_norm): MultiModalityRMSNorm()
            (up_gate_proj): NativeMoELinear()
            (down_proj): NativeMoELinear()
            )
        )
        )
    )
    (final_norm_video): MultiModalityRMSNorm()
    (final_norm_audio): MultiModalityRMSNorm()
    (final_linear_video): Linear(in_features=5120, out_features=192, bias=False)
    (final_linear_audio): Linear(in_features=5120, out_features=64, bias=False)
    )
        '''
    # print_model_size(
    #     model, prefix=f"(tp, cp, pp) rank ({get_tp_rank()}, {get_cp_rank()}, {get_pp_rank()}): ", print_func=print_rank_0
    # )

    # --- FP8 INJECTION FIX: PREPARE MODEL FOR SCALES ---
    for name, module in model.named_modules():
        if module.__class__.__name__ in ['BaseLinear', 'NativeMoELinear']:
            # Uses LEGAL PyTorch naming
            module.register_buffer('weight_scale', torch.tensor(1.0, dtype=torch.float32))

    # --- ORIGINAL LOADER ---
    model = load_model_checkpoint(model, engine_config)

    # --- FP8 INJECTION FIX: SAFE TYPE CASTING ---
    for param in model.parameters():
        if getattr(param, "dtype", None) != torch.float8_e4m3fn:
            param.data = param.data.to(torch_type)
            
    for name, buf in model.named_buffers():
        if getattr(buf, "dtype", None) != torch.float8_e4m3fn and 'weight_scale' not in name:
            buf.data = buf.data.to(torch_type)
    # -----------------------------------------------------------

    if offload:
        model.cuda(torch.cuda.current_device())
    model.eval()
    print_mem_info_rank_0("Load model successfully")

    gc.collect()
    torch.cuda.empty_cache()
    return model
