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

import io
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import gc
from ...common import EngineConfig
from ...utils import print_rank_0
from safetensors.torch import load as load_from_bytes
from safetensors.torch import load_file
from tqdm.auto import tqdm


def _load_shard(shard_path, param_names, num_threads=None):
    zstd_path = shard_path + ".zst"
    if os.path.exists(zstd_path):
        cmd = ["zstd", "-d"]
        if num_threads:
            cmd.extend(["-T", str(num_threads)])  # set parallelism

        process = subprocess.Popen(cmd + ["-c", zstd_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)

        decompressed_data = process.stdout.read()
        while True:
            new_data = process.stdout.read()
            if not new_data:
                break
            decompressed_data += new_data
        process.stdout.close()

        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"Decompression failed: {process.stderr.read().decode()}")

        buffer = io.BytesIO(decompressed_data)
        weights = load_from_bytes(buffer.getvalue())
        buffer.close()
    else:
        weights = load_file(shard_path)

    return {name: weights[name] for name in param_names}


def load_sharded_safetensors_parallel_with_progress(checkpoint_dir):
    # If it's just one file, load it normally
    if os.path.isfile(checkpoint_dir):
        return load_file(checkpoint_dir)

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    
    # If no index, try to find a single model file
    if not os.path.exists(index_path):
        model_file_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(model_file_path):
            return load_file(model_file_path)
        # Fallback for your custom merged filename
        final_file = os.path.join(checkpoint_dir, "davinci_distilled_FINAL.safetensors")
        if os.path.exists(final_file):
            return load_file(final_file)
        raise FileNotFoundError(f"Could not find model files in {checkpoint_dir}")

    with open(index_path, "r") as f:
        index = json.load(f)

    state_dict = {}
    # Get unique shard files from the weight map
    shard_files = sorted(list(set(index["weight_map"].values())))
    
    print(f"Loading {len(shard_files)} shards sequentially to save RAM...")
    
    # We load them ONE BY ONE instead of using multi-threading
    for shard_file in tqdm(shard_files, desc="RAM-Safe Loading"):
        shard_path = os.path.join(checkpoint_dir, shard_file)
        
        # Get the list of parameters that belong to this specific shard
        param_names = [k for k, v in index["weight_map"].items() if v == shard_file]
        
        # Load the shard data
        weights = load_file(shard_path)
        
        # Extract only what we need and update the main dict
        for name in param_names:
            state_dict[name] = weights[name]
        
        # Clean up temporary weights immediately to free memory for the next shard
        del weights
        gc.collect()

    return state_dict


def load_model_checkpoint(model, engine_config: EngineConfig):
    print_rank_0("Loading checkpoint sequentially...")
    state_dict = load_sharded_safetensors_parallel_with_progress(engine_config.load)
    
    # Use assign=True to help with memory, then immediately nuke the dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    
    # Hard wipe the temporary dictionary immediately
    state_dict.clear() 
    del state_dict
    gc.collect()
    
    print_rank_0(f"Load Weight Missing Keys: {missing_keys}")
    print_rank_0(f"Load Weight Unexpected Keys: {unexpected_keys}")
    return model
