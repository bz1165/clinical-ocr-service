# scripts/finetune_lora.py

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

class DataCollatorForCausalLM:
    """
    Pads `input_ids` with tokenizer.pad_token_id and `labels` with -100
    to the max length in the batch, and builds attention_mask accordingly.
    """
    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        # features: list of dicts {"input_ids": [...], "labels": [...]}
        input_seqs = [f["input_ids"] for f in features]
        label_seqs = [f["labels"]    for f in features]
        max_len = max(len(seq) for seq in input_seqs)

        # pad inputs and labels
        padded_inputs = [
            seq + [self.pad_token_id] * (max_len - len(seq))
            for seq in input_seqs
        ]
        padded_labels = [
            seq + [self.label_pad_token_id] * (max_len - len(seq))
            for seq in label_seqs
        ]

        # attention mask: 1 for real tokens, 0 for pad
        attention_mask = [
            [1]*len(seq) + [0]*(max_len - len(seq))
            for seq in input_seqs
        ]

        return {
            "input_ids":      torch.tensor(padded_inputs,   dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask,   dtype=torch.long),
            "labels":         torch.tensor(padded_labels,    dtype=torch.long),
        }

def parse_args():
    p = argparse.ArgumentParser(description="LoRA-fine-tune InternVL3")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--train_file",       required=True)
    p.add_argument("--output_dir",       default="lora_output")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--lr",          type=float, default=3e-4)
    return p.parse_args()

def main():
    args = parse_args()

    # 1) device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 2) tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)

    # 3) apply LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 4) load & preprocess
    print("Loading data…")
    ds = load_dataset("json", data_files={"train": args.train_file})
    def preprocess(ex):
        # tokenize prompt vs completion
        p_ids = tokenizer(ex["prompt"],     truncation=True, max_length=512).input_ids
        a_ids = tokenizer(ex["completion"], truncation=True, max_length=256).input_ids
        return {
            "input_ids": p_ids + a_ids,
            "labels":    [-100]*len(p_ids) + a_ids
        }
    train_ds = ds["train"].map(preprocess, remove_columns=ds["train"].column_names)

    # 5) collator
    collator = DataCollatorForCausalLM(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100
    )

    # 6) training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_checkpointing=True,
        fp16=False, bf16=False,      # MPS needs full-precision
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        optim="adamw_torch",
        report_to="none",
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # 8) train & save
    print("Starting training…")
    trainer.train()
    trainer.save_model(args.output_dir)
    print("LoRA fine-tuning complete!")

if __name__ == "__main__":
    main()
