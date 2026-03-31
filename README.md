<img width="1843" height="705" alt="Screenshot (79)" src="https://github.com/user-attachments/assets/428700f2-da43-4a9c-a75b-0efa47d2af6a" />



-----

<div align="center">

# REBELS fp8 daVinci-MagiHuman NODES

EXPERIMENTAL! please test for me, my gpu is too small to hold the model 😅

have not tested, just merged the files to match another contributors workflow. this will lessen the requirements for lower vram users. its still a 16gb fp8 model and the SR model to upscale it heavy as well. working on offloading implementation without sage or triton. it will be in the workflow once i test it. i will update the repo with a SEPERATE offloading workflow check back soon.

THIS MODEL WILL NOT RUN WITHOUT MY CUSTOM NODE SET. ITS CURATED FOR THIS MODEL SPECIFICALLY!

t5gemma text encoder gguf goes in "gguf" folder NOT text encoder folder


files are in the repo

fp8 model - Diffusion models folder

text encoder gguf - gguf folder, NOT text encoder folder

wan vae - vae folder

sd audio vae - vae folder

turbo vae - vae folder

sr fp8 model - diffusion models folder


<p align="center">
  <a href="https://plms.ai">SII-GAIR</a> &nbsp;&amp;&nbsp; <a href="https://sand.ai">Sand.ai</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2603.21986-b31b1b.svg)](https://arxiv.org/abs/2603.21986)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-HuggingFace-orange)](https://huggingface.co/spaces/SII-GAIR/daVinci-MagiHuman)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-HuggingFace-yellow)](https://huggingface.co/GAIR/daVinci-MagiHuman)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-ee4c2c.svg)](https://pytorch.org/)

</div>


ComfyUI_MagiHuman
----
[DaVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman):Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model


[based on smthemex's node set: https://github.com/smthemex/ComfyUI_MagiHuman]


Update
----
* REBEL added fp8 scaling support for my model, i also added a bypass of flash attn and tiled vae encoding. not tested because i OOM on my 3070 (8gb vram). just doing my part to the community. https://huggingface.co/realrebelai/DaVinci_MagiHuman_fp8_merges/tree/main


* add layer offload num to fit high vram 大显存 把offload开到你跑得动为止，小显存则从1开始测试，MagiCompiler库不用，但是改了麻烦，还是加了回来


1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/RealRebelAI/ComfyUI_MagiHuman
```
2.requirements  
----

```
pip install -r requirements.txt

#python312以下，注释掉MagiCompiler目录下的pyproject.toml的13行的requires-python = ">=3.12" 再安装
git clone https://github.com/SandAI-org/MagiCompiler.git 
cd MagiCompiler
pip install -r requirements.txt
pip install .
```

3.checkpoints 
----
* dit and TE  [links](https://huggingface.co/realrebelai/DaVinci_MagiHuman_fp8_merges/tree/main)

```
├── ComfyUI/models/
|     ├── diffusion_models/
|        ├──fp8.safetensors
|        ├──1080p_sr_merge_fp8.safetensors
|     ├── vae/
|        ├──sd_audio.safetensors
|        ├──Wan2.2_VAE.pth
|     ├── gguf
|        ├──t5gemma-9b-9b-ul2-Q6_K.gguf

```



We thank the open-source community, and in particular [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Turbo-VAED](https://github.com/hustvl/Turbo-VAED), for their valuable contributions.

## 📄 License

This project is released under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

## 📖 Citation

```bibtex
@misc{davinci-magihuman-2026,
  title   = {Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model},
  author  = {SII-GAIR and Sand.ai},
  year    = {2026},
  url     = {https://github.com/GAIR-NLP/daVinci-MagiHuman}
}
```
