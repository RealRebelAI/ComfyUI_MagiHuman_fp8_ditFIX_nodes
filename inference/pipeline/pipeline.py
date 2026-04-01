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
from typing import Optional, Union

import torch
from PIL import Image

from ..common import EvaluationConfig
from ..model.dit import get_dit, DiTModel
from .video_generate import MagiEvaluator


class MagiPipeline:
    """Pipeline facade for inference."""

    def __init__(
        self,
        model: DiTModel,
        evaluation_config: EvaluationConfig,
        config,
        is_distill: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.evaluation_config = evaluation_config
        self.config = config
        self.sr_model = None
        self.device = device
        self.is_distill = is_distill

    def pre_model(self, sr_mode):
        import comfy.model_management as mm
        print(f"infer {sr_mode}")

        if sr_mode:
            print("Running aggressive memory flush before SR model load...")

            # Offload base model blocks to CPU — keeps the object alive for evaluate()
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'block_gpu_manager'):
                    self.model.block_gpu_manager.unload_all_blocks_to_cpu()

            if hasattr(self, 'evaluator'):
                self.evaluator = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            mm.soft_empty_cache()

            print("Memory flushed. Loading SR model...")
            self.config.engine_config.load = self.evaluation_config.sr_model_path
            self.sr_model = get_dit(
                self.config.sr_arch_config,
                self.config.engine_config,
                torch_type=torch.float8_e4m3fn
            )
            # Correct args: (model, sr_model, evaluation_config, device)
            # BlockGPUManager streams each model's blocks on demand — never both in VRAM
            self.evaluator = MagiEvaluator(
                self.model,
                self.sr_model,
                self.evaluation_config,
                self.device
            )

        else:
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'block_gpu_manager'):
                    self.model.block_gpu_manager.unload_all_blocks_to_cpu()

            gc.collect()
            torch.cuda.empty_cache()
            mm.soft_empty_cache()

            self.sr_model = get_dit(
                self.config.sr_arch_config,
                self.config.engine_config,
                torch_type=torch.float8_e4m3fn
            )
            self.evaluator = MagiEvaluator(
                self.model,
                self.sr_model,
                self.evaluation_config,
                self.device
            )

    def _validate_offline_request(self, prompt: str, save_path_prefix: str):
        if not prompt or not prompt.strip():
            raise ValueError("`prompt` must be a non-empty string.")
        if not save_path_prefix or not save_path_prefix.strip():
            raise ValueError("`save_path_prefix` must be a non-empty string.")

    def run_offline(
        self,
        prompt: str,
        image: Union[str, Image.Image, None],
        audio: Optional[str],
        save_path_prefix: str,
        seed: int = 42,
        seconds: int = 4,
        br_width: int = 480,
        br_height: int = 272,
        sr_width: Optional[int] = None,
        sr_height: Optional[int] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        upsample_mode: Optional[str] = None,
        conds: dict = {},
        sr_mode: bool = False,
        steps: int = 50,
        sr_steps: int = 50,
        offload: bool = False,
        offload_block_num: int = 1,
    ):
        self.pre_model(sr_mode)

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.random.manual_seed(seed)
            latent_video, latent_audio, params = self.evaluator.evaluate(
                prompt,
                image,
                audio,
                seconds=seconds,
                br_width=br_width,
                br_height=br_height,
                sr_width=sr_width,
                sr_height=sr_height,
                br_num_inference_steps=steps,
                sr_num_inference_steps=sr_steps,
                conds=conds,
                offload=offload,
                is_distill=self.is_distill,
                offload_block_num=offload_block_num,
            )

        print("Sampling complete. Nuking DiT models from VRAM for VAE decoding...")
        self.model = None
        self.sr_model = None
        self.evaluator = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("VRAM successfully cleared. Handing off to VAE...")

        return latent_video, latent_audio, params
