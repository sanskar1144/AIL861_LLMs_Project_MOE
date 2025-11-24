import torch
import joblib
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time 

start = time.time()

# --- 1. CONFIGURATION ---
BASE_MODEL_PATH = "/Users/sanskar/Downloads/AI Experts/llama/Llama 3.2 1b"
ROUTER_PIPELINE_PATH = "logreg_tfidf_pipeline.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

EXPERT_ADAPTER_MAP = {
    "depression": "/Users/sanskar/Downloads/AI Experts/llama_1b_finetuned_moe_adapter",           # FIX THIS IF DIFFERENT
    "anxiety": "/Users/sanskar/Downloads/AI Experts/llama_1b_finetuned_moe_adapter_anxiety",
    "schizophrenia": "/Users/sanskar/Downloads/AI Experts/llama_1b_finetuned_moe_adapter_schizophrenia",
    "bpd": "/Users/sanskar/Downloads/AI Experts/llama_1b_finetuned_moe_adapter_bpd",
    "ocd": "/Users/sanskar/Downloads/AI Experts/llama_1b_finetuned_moe_adapter_ocd",
    # "general": "<path to some general adapter or base>",
}

# numeric label → string label (must match your training)
NUM_TO_STRING_MAP = {
    0: "depression",
    1: "anxiety",
    2: "schizophrenia",
    3: "ocd",
    4: "bpd",
}

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Loading system on device: {DEVICE}")

# --- 2. Load router ---
router_pipeline = joblib.load(ROUTER_PIPELINE_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("Router models loaded successfully.")

# --- 3. Load base model + tokenizer ---
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": DEVICE},      # simple map for single device
    trust_remote_code=True,
)
base_model.config.use_cache = True

base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token
base_tokenizer.padding_side = "left"

print("Base Llama model loaded successfully.")

# --- 4. Expert cache ---
EXPERT_MODELS = {}  # expert_label -> PeftModel


def get_expert_model(expert_label: str, expert_path: str):
    """
    Lazily load and cache the PEFT expert model for a given expert_label.
    """
    if expert_label in EXPERT_MODELS:
        return EXPERT_MODELS[expert_label]

    print(f"[get_expert_model] Loading adapter for '{expert_label}' from: {expert_path}")
    expert_model = PeftModel.from_pretrained(
        base_model,
        expert_path,
        is_trainable=False,
    )
    expert_model.to(DEVICE)
    expert_model.eval()
    EXPERT_MODELS[expert_label] = expert_model
    return expert_model


# --- 5. MoE generation ---
def generate_moe_response(prompt: str, max_new_tokens: int = 256) -> tuple[str, str]:
    """
    Runs the prompt through the router to select an expert, then generates a response.
    Returns: (response_text, expert_name)
    """

    # 5.1 Router prediction
    preds_encoded = router_pipeline.predict([prompt])
    expert_index = preds_encoded[0]  # integer class, e.g. 4
    expert_label = NUM_TO_STRING_MAP.get(expert_index, "unknown_class")
    expert_path = EXPERT_ADAPTER_MAP.get(expert_label)

    if not expert_path:
        print(f"WARNING: No adapter path found for label '{expert_label}'. Falling back to base model.")
        expert_model = base_model
        expert_label = "base_model_fallback"
    else:
        expert_model = get_expert_model(expert_label, expert_path)

    print(f"\n--- ROUTER SELECTED: {expert_label} (Adapter: {expert_path}) ---")
    # 5.2 Tokenization
    input_text = f"<s>[INST] {prompt.strip()} [/INST]"

    inputs = base_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    input_ids = inputs["input_ids"]

    # 5.3 Generation
    with torch.no_grad():
        outputs = expert_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=base_tokenizer.eos_token_id,
            pad_token_id=base_tokenizer.eos_token_id,
        )

    generated_ids = outputs[0]

    # 5.4 Stripping off the prompt → keep only new tokens
    gen_only_ids = generated_ids[input_ids.shape[-1]:]

    response_text = base_tokenizer.decode(
        gen_only_ids,
        skip_special_tokens=True
    ).strip()

    full_decoded = base_tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"[FULL DECODED]: {full_decoded!r}")
    print(f"[RESPONSE ONLY]: {response_text!r}")

    return response_text, expert_label


# --- 6. usage ---
if __name__ == "__main__":
    test_prompts = [
        "What are the best methods for managing bipolar disorder and clinical depression?"
    ]

    for i, p in enumerate(test_prompts):
        print("=" * 60)
        print(f"PROMPT {i+1}: {p}")
        response, expert = generate_moe_response(p)
        print(f"\n[ROUTER DECISION]: {expert.upper()}")
        print(f"\n[RESPONSE]: {response}")
        print("=" * 60)
    end = time.time()
    print(f"Total inference time: {end - start} seconds")
