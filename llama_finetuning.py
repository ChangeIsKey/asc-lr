from datasets import load_dataset
import torch
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, AddedToken, EarlyStoppingCallback
from peft import LoraConfig
import os
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

LLMs_CACHE_DIR = os.environ["TMPDIR"]

def formatting_func(batch_of_examples):
    batch_formatted = []
    for i in range(len(batch_of_examples['input'])):
        answer_formatted = '<|s|>'.join(batch_of_examples['output'][i])
        text = f"{batch_of_examples['input'][i]}<|answer|>{answer_formatted}<|end|>"
        batch_formatted.append(text)
    return batch_formatted



def load_data():
    train_dataset = load_dataset('json', data_files='datasets/train.jsonl', split='train')
    eval_dataset = load_dataset('json', data_files='datasets/dev.jsonl', split='train')
    return train_dataset, eval_dataset



def train(base_model_name, qlora):
    base_model_name_mod = base_model_name.split('/')[1]

    if qlora:
        print('QLORA')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )


        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True,
            cache_dir=LLMs_CACHE_DIR
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True,
            cache_dir=LLMs_CACHE_DIR
        )
    base_model.config.use_cache = False

    base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False, trust_remote_code=True, cache_dir=LLMs_CACHE_DIR)
    tokenizer.add_special_tokens({ "additional_special_tokens":[AddedToken("<|s|>"), AddedToken("<|answer|>"), AddedToken("<|end|>")]})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'left'
    base_model.resize_token_embeddings(len(tokenizer))

    max_seq_length = 512

    def tokenize(element):
        outputs = tokenizer(
            formatting_func(element),
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    train_dataset, eval_dataset = load_data()
    output_dir = 'models/qlora_'+base_model_name_mod

    batch_size = 16

    response_template = "<|answer|>"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)



    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=50,
        num_train_epochs=10,
        max_steps=-1,
        logging_dir=f"logs/logs_{base_model_name_mod}",  # Directory for storing logs
        save_strategy="epoch",  # Save the model checkpoint every logging step
    )


    if qlora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            modules_to_save=["embed_tokens", "lm_head"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            #eval_dataset=eval_dataset,
            peft_config=peft_config,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            data_collator=collator,
            args=training_args
        )
    else:
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            data_collator=collator,
            args=training_args,
        )

    # pass in resume_from_checkpoint=True to resume from a checkpoint
    trainer.train()


if __name__ == '__main__':
    base_model_name =  sys.argv[1]
    qlora = int(sys.argv[2])
    train(base_model_name, qlora)





