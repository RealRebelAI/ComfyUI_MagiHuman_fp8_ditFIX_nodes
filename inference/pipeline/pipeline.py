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

from typing import Optional, Union
import torch
from PIL import Image
from ..common import EvaluationConfig
from ..model.dit import get_dit
from ..model.dit import DiTModel
from .video_generate import MagiEvaluator



class MagiPipeline:
    """Pipeline facade for inference."""

    def __init__(self, model: DiTModel, evaluation_config: EvaluationConfig, config,is_distill=True,device: str = "cuda"):
        self.model = model
        self.evaluation_config = evaluation_config
        self.config = config
        self.sr_model=None
        self.device = device
        self.is_distill=is_distill
        # if self.evaluation_config.use_sr_model:
        #     config.engine_config.load = evaluation_config.sr_model_path
        #     sr_model = get_dit(config.sr_arch_config, config.engine_config)
        #     self.model=None
        # else:
        #     sr_model = None
       
    
    def pre_model(self, sr_mode):
        import gc
        print(f"infer {sr_mode}")
        
        if sr_mode:
            print("Running aggressive memory flush before SR model load...")
            # 1. Unlink the massive base model
            self.model = None
            if hasattr(self, 'evaluator'):
                self.evaluator = None
                
            # 2. Force Python to run garbage collection (clears System RAM)
            gc.collect()
            
            # 3. Force PyTorch to empty the VRAM cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            print("Memory flushed. Loading SR model...")
            
            # Now it is safe to load the massive SR model
            self.config.engine_config.load = self.evaluation_config.sr_model_path
            self.sr_model = get_dit(self.config.sr_arch_config, self.config.engine_config, torch_type=torch.float8_e4m3fn)
            self.evaluator = MagiEvaluator(self.model, self.sr_model, self.evaluation_config, self.config, self.device)
            
        else:
            self.sr_model = get_dit(self.config.sr_arch_config, self.config.engine_config, torch_type=torch.float8_e4m3fn)
            self.evaluator = MagiEvaluator(self.model, self.sr_model, self.evaluation_config, self.config, self.device)


    def _validate_offline_request(
        self,
        prompt: str,
        save_path_prefix: str,
    ):
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
        conds={},
        sr_mode=False,
        steps=50,
        sr_steps=50,
        offload=False,
        offload_block_num=1,


    ):
        #self._validate_offline_request(prompt=prompt, save_path_prefix=save_path_prefix)

        # if self.evaluator.sr_model is not None:
        #     save_path = f"{save_path_prefix}_{seconds}s_{br_width}x{br_height}_{sr_width}x{sr_height}.mp4"
        # else:
        #     save_path = f"{save_path_prefix}_{seconds}s_{br_width}x{br_height}.mp4"

        self.pre_model(sr_mode)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.random.manual_seed(seed)
            latent_video, latent_audio,params = self.evaluator.evaluate(
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

        # if output_width is not None and output_height is not None:
        #     video_np = upsample_video(video_np, output_width, output_height, upsample_mode)

        # if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
        #     saving_name = f"{prompt.replace(' ', '_')[:10]}"
        #     audio_path = saving_name + str(random.randint(0, 1000000)) + ".wav"
        #     video_path = saving_name + str(random.randint(0, 1000000)) + ".mp4"
        #     sf.write(audio_path, audio_np, self.evaluator.audio_vae.sample_rate)
        #     imageio.mimwrite(video_path, video_np, fps=self.evaluation_config.fps, quality=8, output_params=["-loglevel", "error"])
        #     assert os.path.exists(video_path)
        #     merge_video_and_audio(video_path, audio_path, save_path)

        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()

        print("Sampling complete. Nuking DiT models from VRAM for VAE decoding...")
        self.model = None
        self.sr_model = None
        self.evaluator = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("VRAM successfully cleared. Handing off to VAE...")

        return latent_video, latent_audio,params

