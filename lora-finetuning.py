#!pip3 install -q bitsandbytes datasets accelerate loralib
#!pip3 install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git


#Model Loading 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-1.3B", 
    load_in_8bit=True, 
    device_map={"": "gpu"},
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B") #decapoda-research/llama-13b-hf

#Post-processing on the model
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)


#Applying Lora
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model 

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)



"""Reading the Data"""

# Read the data from the Excel file
data_df = pd.read_excel("Juice Wrld dataset.xlsx")

# Convert the data to a DatasetDict object
data_dict = DatasetDict({
    "train": Dataset.from_dict({
        "quote": [f"{data_df.iloc[1][0]} [{data_df.iloc[0][0]}] {data_df.iloc[1][1]}"],
        "author": [data_df.iloc[1][0]]
    })
})

del data_dict["train"].features["tags"]


data = data_dict



tokenizer.add_special_tokens({'pad_token': '[PAD]'})



#Training the model

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


model.push_to_hub("Amirkid/optm2", use_auth_token=True)
