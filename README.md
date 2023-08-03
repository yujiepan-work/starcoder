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

codegen25-7b

```bash
export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=0

folder=0803_codegen25-7b-multi_naive_chunks
mkdir ./logs/$folder

torchrun --nproc_per_node=2 finetune/finetune_bios.py \
  --model_path="Salesforce/codegen25-7b-multi" \
  --lora_target_modules=codegen-7 \
  --dataset_name=naive_chunks \
  --naive_chunks_path=/mnt/sh_flex_storage/home/yujiepan/workspace/bios-llm/starcoder/data/ipsd_data1_by_codegen25.pt \
  --seq_length 2048 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_steps 4000 \
  --eval_freq 50 \
  --save_freq 50 \
  --log_freq 5 \
  --learning_rate 5e-5 \
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 50 \
  --weight_decay 0.05\
  --run_name=$folder \
  --output_dir="./logs/$folder" 2>&1 | tee ./logs/$folder/std.log
```