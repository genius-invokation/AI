import json
# import pandas as pd
import torch
# from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from torch.nn import DataParallel
import wandb
wandb.init(project="gitcg", name="debugtryforhead")

from torch.utils.data import Dataset, DataLoader, IterableDataset

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
# tokenizer.pad_token_id = tokenizer.eos_token_id
# Check if pad_token is already set
# if tokenizer.pad_token is None:
    # Add a new pad token
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    # print("Added pad_token:", tokenizer.pad_token)
# else:
    # print("Existing pad_token:", tokenizer.pad_token)

class StreamingDataset(IterableDataset):
    def __init__(self, get_data):
        self.get_data = get_data
        self.want_to_stop = False
    def process_func(self, example):
        # text = tokenizer.apply_chat_template(
            # example[0], tokenize=False, add_generation_prompt=True
        # )
        text = example[0]
        input_ids = tokenizer(text)["input_ids"] # first try no padding
        labels = torch.tensor(example[1], dtype=torch.bfloat16) 
        return {"input_ids": input_ids, "labels": labels}

    def __iter__(self):
        while True:
            data = self.get_data()
            yield self.process_func(data)

# dataset = StreamingDataset(get_batch)

def prepare_model(run_mode):
    if run_mode == "train_loss_head":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,device_map="auto", torch_dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        
        for name, param in model.named_parameters():
            print(name)
            if name == "score.weight":
                param.requires_grad = True
            else:
                param.requires_grad = False
        # print(model.score) is Linear(in_features=896, out_features=1, bias=False)

    elif run_mode == "train_rl": 
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.bfloat16)
        """
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        """
        loaded_state_dict = torch.load('score_layer_weights.pth')
        model.score.load_state_dict(loaded_state_dict)
        model.config.pad_token_id = tokenizer.pad_token_id

        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            inference_mode=False,  # 训练模式
            r=64,  # Lora 秩
            lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=0.1,  # Dropout 比例
        )

        model = get_peft_model(model, config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallel(model).to(device)
    elif run_mode == "generate":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
    return model

default_model = prepare_model("generate")

def get_lm_response(prompt):

    model = default_model
    # messages = [
    #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
    model_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response

def reason(prompts, model):
    """
    Handles multiple prompts and returns a list of regression values.

    Args:
        prompts (str or List[str]): Input text prompts.
        model: A regression model (e.g., AutoModelForSequenceClassification with num_labels=1).

    Returns:
        List[float]: Regression values for each input prompt.
    """
    # Static tokenizer initialization (singleton pattern)
    if not hasattr(reason, "tokenizer"):
        reason.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    tokenizer = reason.tokenizer
    device = model.module.device if hasattr(model, "module") else model.device
    # Tokenize the input prompts and move to the model's device
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)#(model.device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(**model_inputs)  # Forward pass

    # Extract logits (regression outputs)
    logits = outputs.logits  # Shape: [batch_size, 1]

    # Convert logits to a list of floats
    regression_values = logits.squeeze(-1).tolist()  # Remove the extra dimension
    print("regression_values", regression_values)
    return regression_values



def train(train_dataset, run_mode, model=None):
    if model is None:
        model = prepare_model(run_mode)
    if run_mode == "train_loss_head":
        args = TrainingArguments(
            output_dir=f"./output/{model_name}",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=1,
            # save_steps=15,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            report_to="wandb",
            max_steps=60,
            # load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        
    elif run_mode == "train_rl": 
        args = TrainingArguments(
            output_dir=f"./output/{model_name}",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=1,
            save_steps=15,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_steps=5,
            report_to="wandb",
            max_steps=30,
            # load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        """
        args = TrainingArguments(
            output_dir=f"./output/{model_name}",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=10,
            num_train_epochs=1,
            save_steps=20,
            learning_rate=1e-4,
            report_to="wandb",
            max_steps=200,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        """
    trainer.train()
    if run_mode == "train_loss_head":
        torch.save(model.score.state_dict(), 'score_layer_weights.pth')
    else:
        model.save_pretrained("save_model")
    train_dataset.want_to_stop = True



        
# def process_func(example):
#     """
#     将数据集进行预处理
#     """
#     MAX_LENGTH = 384
#     input_ids, attention_mask, labels = [], [], []
#     instruction = tokenizer(
#         f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
#         add_special_tokens=False,
#     )
#     response = tokenizer(f"{example['output']}", add_special_tokens=False)
#     input_ids = (
#         instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
#     )
#     attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
#     labels = (
#         [-100] * len(instruction["input_ids"])
#         + response["input_ids"]
#         + [tokenizer.pad_token_id]
#     )
#     if len(input_ids) > MAX_LENGTH:  
#         breakpoint()
#         input_ids = input_ids[:MAX_LENGTH]
#         attention_mask = attention_mask[:MAX_LENGTH]
#         labels = labels[:MAX_LENGTH]
        
#     return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response
