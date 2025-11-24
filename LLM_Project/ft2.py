import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Define the sequence length once
MAX_SEQ_LENGTH = 1024 

# --- 0. Configuration and Environment Setup ---
# Set the device to MPS for Apple Silicon GPU acceleration
if not torch.backends.mps.is_available():
    print("WARNING: MPS not available. Falling back to CPU. Training will be extremely slow.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("mps")

# Set seed for reproducibility
set_seed(42)

# --- 1. Model & Data Paths (MUST BE ADJUSTED) ---

MODEL_PATH = "/Users/sanskar/Downloads/AI Experts/llama/Llama 3.2 1b"

# You will have to uncomment the file for which you want to fine-tune the model.
    
DATASET_FILE_PATH = "./data_depression.json" 
OUTPUT_DIR = "./llama_1b_finetuned_moe_adapter"

# DATASET_FILE_PATH = "./data_anxiety.json" 
# OUTPUT_DIR = "./llama_1b_finetuned_moe_adapter_anxiety"

# DATASET_FILE_PATH = "./data_schizophrenia.json" 
# OUTPUT_DIR = "./llama_1b_finetuned_moe_adapter_schizophrenia"

# DATASET_FILE_PATH = "./data_bpd.json" 
# OUTPUT_DIR = "./llama_1b_finetuned_moe_adapter_bpd"

# DATASET_FILE_PATH = "./data_ocd.json" 
# OUTPUT_DIR = "./llama_1b_finetuned_moe_adapter_ocd"


# --- 2. Load Model & Tokenizer with MPS Settings ---

print(f"Loading model from: {MODEL_PATH} onto device: {DEVICE}")

# Load the model in bfloat16 (preferred on M-series)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16, # Fixed: Using 'dtype' instead of 'torch_dtype'
    device_map=DEVICE, # Map the model to the Mac's GPU
    trust_remote_code=True,
)
model.config.use_cache = False

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# --- 3. LoRA Configuration (PEFT) ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32, 
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)
print("\n--- Trainable Parameters ---")
model.print_trainable_parameters()
print("--------------------------\n")

# --- 4. Load Custom JSON Dataset and Format (MANUAL PRE-PROCESSING) ---

def formatting_func(example):
    """
    Formats the instruction, input, and output fields into the Llama instruction template.
    Adds the formatted text string to a new 'text' column in the example dictionary.
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    response = example.get("output", "")
    
    # Combine instruction and optional input
    if input_text and input_text.strip():
        instruction_content = f"{instruction}\n\n[CONTEXT]: {input_text}"
    else:
        instruction_content = instruction

    # The full instruction-response pair in Llama's chat format: <s>[INST] Instruction [/INST] Response</s>
    text = (
        f"<s>[INST] {instruction_content.strip()} [/INST] "
        f"{response}</s>"
    )
    example["text"] = text
    return example

# Load the local JSON file.
dataset = load_dataset('json', data_files=DATASET_FILE_PATH, split="train")

print("Applying formatting and tokenization manually to dataset...")
# This creates the necessary "text" column.
processed_dataset = dataset.map(
    formatting_func,
    # Remove old columns (instruction, input, output) to keep only the "text" column
    remove_columns=dataset.column_names, 
    batched=False
)
print(f"Dataset columns after processing: {processed_dataset.column_names}")


# --- 5. Training Arguments (MPS Optimized) ---
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=1, 
    optim="adamw_torch", 
    logging_steps=10,
    learning_rate=2e-4,
    num_train_epochs=1, 
    fp16=False,
    bf16=True, 
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    report_to="none", 
)


# --- 6. Initialize and Run SFTTrainer (CLEANED UP) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset, 
    peft_config=peft_config,
    # max_seq_length=MAX_SEQ_LENGTH, # Define this as a constant at the top
    # tokenizer=tokenizer,
    args=training_arguments,
    # formatting_func=formatting_func, 
    # dataset_text_field="text",
)

print("\n--- Starting Training on MPS ---")
trainer.train()

# --- 7. Save the Fine-Tuned LoRA Adapter ---
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n Fine-tuning complete! Adapter saved to: {OUTPUT_DIR}")