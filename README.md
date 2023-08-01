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
conda env config vars set HF_METRICS_CACHE="/mnt/sh_flex_storage/projects/huggingface_home/metrics/"
conda env config vars set HF_EVALUATE_CACHE="/mnt/sh_flex_storage/projects/huggingface_home/evaluate/"
# after re-activate:
conda env config vars list
```

## try out finetuning
```bash
export CUDA_VISIBLE_DEVICES=0
# avoid connections to hf 
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_OFFLINE=1
python finetune/finetune_debug.py \
  --model_path="bigcode/starcoderbase-7b" \
  --dataset_name=/mnt/sh_flex_storage/dataset/hf_datasets/enoreyes--success-llm-instructions/ \
  --size_valid_set 10000\
  --seq_length 256 \
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="prompt"\
  --output_column_name="completion"\
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```