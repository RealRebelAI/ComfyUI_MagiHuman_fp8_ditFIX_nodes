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
    from safetensors.torch import load_file
    import torch
    import gc

    # 1. Handle Single File Case
    if os.path.isfile(checkpoint_dir):
        # mmap=False stops the "Access Violation" on Windows 16GB systems
        return load_file(checkpoint_dir, device="cpu", mmap=False)

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    state_dict = {}
    shard_files = sorted(list(set(index["weight_map"].values())))
    print(f"REBEL AI MODE: Streaming {len(shard_files)} shards without mmap...")

    for shard_file in tqdm(shard_files, desc="RAM-Safe Load"):
        shard_path = os.path.join(checkpoint_dir, shard_file)
        param_names = [k for k, v in index["weight_map"].items() if v == shard_file]
        
        # Loading with mmap=False prevents the "deallocated bytearray" error
        current_shard = load_file(shard_path, device="cpu", mmap=False)
        
        for name in param_names:
            # .clone() makes a permanent copy so we can safely delete the shard
            state_dict[name] = current_shard[name].clone()
        
        del current_shard
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
