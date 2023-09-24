# !pip install -q huggingface_hub
# !pip install -q -U trl transformers accelerate peft
# !pip install -q -U datasets bitsandbytes einops wandb
from auto_gptq import exllama_set_max_input_length

import wandb
# Uncomment to install new features that support latest models like Llama 2
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers.git

# When prompted, paste the HF access token you created earlier.


from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# dataset_name = "<your_hf_dataset>"
# dataset = load_dataset(dataset_name, split="train")

base_model_name = "/home/markmiller/text-generation-webui/models/TheBloke_Wizard-Vicuna-30B-Uncensored-GPTQ_gptq-4bit-32g-actorder_True"

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class ChatDataset(Dataset):
    def __init__(self, json_path, tokenizer: PreTrainedTokenizerFast, max_length=2048):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        output = item['output']

        chat_text = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\nUSER: {instruction}\nASSISTANT: {output}"


        tokenized = self.tokenizer(chat_text, truncation=True, padding='max_length', max_length=self.max_length - 1, return_tensors="pt")

        return {"input_ids": tokenized.input_ids.squeeze(0), "attention_mask": tokenized.attention_mask.squeeze(0)}




# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=device_map
)
base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = ChatDataset('chat.json', tokenizer)

output_dir = "./results"

base_model = exllama_set_max_input_length(base_model, 2048)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500
)

max_seq_length = 2048

wandb.login(key="806e53e3399a2da7b6c327f6d7a84cd81176275b")

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)