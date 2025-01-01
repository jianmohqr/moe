#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers.hf_argparser import HfArgumentParser

import deepspeed


@dataclass
class ModelArguments:
    """
    模型相关的参数
    """
    model_name_or_path: str = field(
        default="mistralai/Mixtral-8x7B-v0.1",
        metadata={"help": "预训练模型名称或本地路径"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "若与 model_name_or_path 不同，可在此指定 tokenizer 名称或路径"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用 fast tokenizer"}
    )


@dataclass
class DataTrainingArguments:
    """
    数据集和训练相关的参数
    """
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "训练集文件（json 或其他格式）"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "验证集文件（可选）"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "输入序列的最大长度"}
    )


@dataclass
class OptimizationArguments:
    """
    优化器相关的参数
    - 这里将原先的 num_train_epochs 和 learning_rate 分别更名为 my_num_train_epochs 和 my_learning_rate
    """
    my_num_train_epochs: int = field(
        default=1,
        metadata={"help": "训练 epoch 数（原 num_train_epochs 更名为 my_num_train_epochs）"}
    )
    my_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "初始学习率（原 learning_rate 更名为 my_learning_rate）"}
    )
    my_per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "每张 GPU 上的 batch size"}
    )
    my_gradient_accumulation_steps: int = field(
        default=32,
        metadata={"help": "梯度累积步数"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用 LoRA 进行微调"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank 大小"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha 参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout 概率"}
    )


def main():
    # === 解析命令行参数 ===
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments))
    model_args, data_args, training_args, optim_args = parser.parse_args_into_dataclasses()

    # === 打印可用 GPU 信息 ===
    print("\n=== GPU 信息 ===")
    print(f"可用的 GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, 显存: {props.total_memory / 1024**3:.1f} GB")
    print("================\n")

    # === 加载 Tokenizer ===
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right",
        cache_dir="./cache",        
        resume_download=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 加载预训练模型 (8-bit + BF16 + 自动多卡) ===
    print("开始加载模型（可能需要一段时间）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=True,               
        torch_dtype=torch.bfloat16,      
        #device_map="auto",               
        low_cpu_mem_usage=True,
        cache_dir="./cache",
        resume_download=True
    )
    print("模型加载完毕!")

    # === 如果要使用 LoRA (PEFT) ===
    if optim_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=optim_args.lora_rank,
            lora_alpha=optim_args.lora_alpha,
            lora_dropout=optim_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj"]  # 仅供示例，不同模型可能模块名不同
        )
        model = get_peft_model(model, lora_config)
        print(f"已应用 LoRA，rank={optim_args.lora_rank}, alpha={optim_args.lora_alpha}, dropout={optim_args.lora_dropout}")

    # === DeepSpeed 配置 ===
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            }
        },
        "train_batch_size": optim_args.my_per_device_train_batch_size * torch.cuda.device_count() * optim_args.my_gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": optim_args.my_per_device_train_batch_size,
        "gradient_accumulation_steps": optim_args.my_gradient_accumulation_steps,
        "bf16": {"enabled": True},
        "zero_allow_untested_optimizer": True
    }
    training_args.deepspeed = ds_config

    # === 将自定义的字段赋值给 TrainingArguments，以避免冲突 ===
    training_args.num_train_epochs = optim_args.my_num_train_epochs
    training_args.learning_rate = optim_args.my_learning_rate
    training_args.per_device_train_batch_size = optim_args.my_per_device_train_batch_size
    training_args.gradient_accumulation_steps = optim_args.my_gradient_accumulation_steps
    training_args.report_to = []

    # === 加载数据集 ===
    if data_args.train_file is not None:
        try:
            dataset = load_dataset(
                "json",
                data_files={"train": data_args.train_file},
                cache_dir="./cache"
            )
            print(f"已加载自定义训练数据: {data_args.train_file}")
        except Exception as e:
            print(f"加载 {data_args.train_file} 失败，报错：{e}\n改为使用 wikitext-2 数据集...")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        print("未指定 train_file，使用 wikitext-2 进行示例训练")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True
        )

    column_names = dataset["train"].column_names
    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names
    )

    # === 创建数据整理器 ===
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # === 创建 Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset.get("validation", None),
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()
    print("训练完成!")

    # === 保存最终模型 (含 LoRA 权重) ===
    trainer.save_model()
    print(f"模型已保存到: {training_args.output_dir}")


if __name__ == "__main__":
    main()
