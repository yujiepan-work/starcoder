from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from pathlib import Path

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="bigcode/large-model")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    return parser.parse_args()

def main():
    args = get_args()

    print('Loading model...')
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16 
    )

    print('Merging model...')
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)

    if args.push_to_hub and False:  # never upload to hf
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
        tokenizer.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    else:
        print('Saving model...')
        save_path = Path(args.peft_model_path).parent / (Path(args.peft_model_path).name + "-merged")
        save_path = save_path.expanduser().resolve().as_posix()
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__" :
    main()
