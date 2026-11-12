"""QLoRA/LoRA fine-tuning entry point for Llama 3.3 70B Instruct.

Install in a CUDA/Linux environment with adequate GPU memory:
  pip install "torch>=2.4" transformers datasets peft trl accelerate bitsandbytes

Run separately for planner and executor:
  python train_lora.py --data data/synthetic/planner_train.jsonl --output adapters/planner
  python train_lora.py --data data/synthetic/executor_train.jsonl --output adapters/executor

Llama 3.3 70B is a large model: 4-bit QLoRA still requires substantial GPU memory
and multi-GPU sharding in most setups. This script does not make 70B training viable
on an ordinary laptop. Use an authorized Hugging Face model access token.
"""
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--model",default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--epochs",type=float,default=2)
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--grad-accum",type=int,default=16)
    ap.add_argument("--max-length",type=int,default=2048)
    ap.add_argument("--lr",type=float,default=2e-4)
    args=ap.parse_args()
    if not Path(args.data).is_file():
        ap.error(f"Training data file does not exist: {args.data}")
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        ap.error(f"Missing optional LoRA training dependencies: {exc}. Install requirements-lora.txt")
    if not torch.cuda.is_available():
        ap.error("CUDA GPU is required for the 70B QLoRA training configuration; CPU training is not supported by this script.")
    ds=load_dataset("json",data_files=args.data,split="train")
    if len(ds)==0 or "messages" not in ds.column_names:
        ap.error("Expected non-empty JSONL chat data with a 'messages' field.")
    for i,row in enumerate(ds):
        if not isinstance(row["messages"],list) or len(row["messages"])<2:
            ap.error(f"Invalid chat example at row {i}; expected at least user and assistant messages.")
    tokenizer=AutoTokenizer.from_pretrained(args.model,use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype,
                             bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(
        args.model,quantization_config=quant,device_map="auto",
        torch_dtype=compute_dtype)
    model=prepare_model_for_kbit_training(model)
    peft=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
                    bias="none",task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj",
                                    "gate_proj","up_proj","down_proj"])
    def format_row(row):
        return tokenizer.apply_chat_template(row["messages"],tokenize=False,
                                             add_generation_prompt=False)
    ds=ds.map(lambda row:{"text":format_row(row)})
    cfg=SFTConfig(output_dir=args.output,num_train_epochs=args.epochs,
                  per_device_train_batch_size=args.batch_size,
                  gradient_accumulation_steps=args.grad_accum,
                  learning_rate=args.lr,logging_steps=10,
                  save_strategy="epoch",bf16=(compute_dtype==torch.bfloat16),fp16=(compute_dtype==torch.float16),
                  gradient_checkpointing=True,max_length=args.max_length,
                  dataset_text_field="text",packing=False,
                  report_to="none")
    trainer=SFTTrainer(model=model,args=cfg,train_dataset=ds,
                       processing_class=tokenizer,peft_config=peft)
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Saved LoRA adapter:",args.output)

if __name__=="__main__": main()
