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

import torch
import gc
from comfy_api.latest import io


class MagiHuman_UnloadDiT(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MagiHuman_UnloadDiT",
            display_name="MagiHuman Unload DiT",
            category="MagiHuman_SM",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        # Delete main DiT only — sr_model, vae, audio_vae are kept intact
        if hasattr(model, 'model') and model.model is not None:
            del model.model
            model.model = None

        torch.cuda.empty_cache()
        gc.collect()
        print("[MagiHuman] Main DiT unloaded. RAM freed for SR model.")
        return io.NodeOutput(model)


NODE_CLASS_MAPPINGS = {"MagiHuman_UnloadDiT": MagiHuman_UnloadDiT}
NODE_DISPLAY_NAME_MAPPINGS = {"MagiHuman_UnloadDiT": "MagiHuman Unload DiT"}
