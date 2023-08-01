## setup

```bash
conda create -n bios-llm python=3.10
conda activate bios-llm
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft.git
pip install tiktoken
pip install bitsandbytes
pip install wandb
pip install scipy

# log in to hf & wandb
huggingface-cli login
wandb login
```

## env var.
```bash
conda env config vars set HF_DATASETS_CACHE="/mnt/sh_flex_storage/projects/huggingface_home/datasets/"
conda env config vars set HUGGINGFACE_HUB_CACHE="/mnt/sh_flex_storage/projects/huggingface_home/hub/"
# after re-activate:
conda env config vars list
```