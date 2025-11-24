# LLM Project – Mixture-of-Experts Mental Health Assistant

This repository contains a full end-to-end pipeline for building a **Mixture-of-Experts (MoE)** LLaMA-3 based mental-health question-answering system.

The system uses:

* **Five fine-tuned LoRA experts** (one per domain)
* A **logistic-regression router** and a **Naive Bayes Model** trained on TF-IDF features
* A unified **MoE inference engine** that dynamically selects the correct expert

Each expert specializes in a specific mental-health domain:
**Depression, Anxiety, Bipolar Disorder (BPD), OCD, and Schizophrenia.**

---

##  Project Structure

```
LLM_PROJECT/
│
├── data_anxiety.json
├── data_bpd.json
├── data_depression.json
├── data_ocd.json
├── data_router.json
├── data_schizophrenia.json
│
├── ft2.py
├── router_lr.py
├── router_nbs.py
├── moe2.py
│
├── label_encoder.joblib
├── logreg_tfidf_pipeline.joblib
│
└── readme.md
```

Below is the meaning of every file.

---

##  Data Files

### **`data_anxiety.json`**

Dataset for training the **Anxiety** expert. Manually created from the data.

### **`data_bpd.json`**

Dataset for **Bipolar Disorder (BPD)** expert. Manually created from the data.

### **`data_depression.json`**

Dataset for **Depression** expert. Manually created from the data.

### **`data_ocd.json`**

Dataset for **Obsessive-Compulsive Disorder** expert. Manually created from the data.

### **`data_schizophrenia.json`**

Dataset for **Schizophrenia** expert. Manually created from the data.

### **`data_router.json`**

Training dataset for the **router classifier**. Contains:

* Prompt text
* Integer class label (0–4)

This trains the logistic regression/ Naive Bayes model that decides which expert to use.

---

## Fine-Tuning

### **`ft2.py` — LLaMA Fine-Tuning (LoRA)**

This script performs LoRA-based fine-tuning for **one domain at a time**.

It does:

* Loads LLaMA-3.2-1B base model
* Prepares instruction-response pairs
* Applies LoRA (PEFT)
* Fine-tunes the model using TRL’s `SFTTrainer`
* Saves the LoRA adapter
> Run this script 5 times — once per domain.

---

## Router Training

### **`router_lr.py` — Train the Gating Router**

This script trains the TF-IDF + Logistic Regression pipeline that predicts the domain.

Outputs:

* **`logreg_tfidf_pipeline.joblib`** – router model
* **`label_encoder.joblib`** – integer↔label mapping

### **`router_nbs.py` — Train the Gating Router**

This script trains the TF-IDF + Logistic Regression pipeline that predicts the domain.

Outputs:

* **`mnb_tfidf_pipeline.joblib`** – router model
* **`label_encoder.joblib`** – integer↔label mapping

These files are used by `moe2.py` during inference.

---

## MoE Inference Engine

### **`moe2.py`** — Core MoE Pipeline

Responsible for:

1. Loading the base LLaMA model
2. Loading all expert LoRA adapters
3. Using the router to select the correct LoRA expert
4. Generating the final model response

Pipeline flow:

```
User Query → Router → Expert Selection → LLaMA + LoRA Expert → Response
```

---

## Router Model Artifacts

### **`logreg_tfidf_pipeline.joblib`**

The trained TF-IDF + Logistic Regression pipeline.

### **`label_encoder.joblib`**

Maps:

* `0 → depression`
* `1 → anxiety`
* `2 → schizophrenia`
* `3 → ocd`
* `4 → bpd`

---

## How to Train Experts

Run fine-tuning separately for each domain by editing dataset path inside `ft2.py`:

```bash
python ft2.py
```

This will generate 5 adapter folders:

```
llama_1b_finetuned_moe_adapter_depression/
llama_1b_finetuned_moe_adapter_anxiety/
llama_1b_finetuned_moe_adapter_bpd/
llama_1b_finetuned_moe_adapter_ocd/
llama_1b_finetuned_moe_adapter_schizophrenia/
```

---

## Train Router

```bash
python router.py
```

This creates:

* `logreg_tfidf_pipeline.joblib`
* `label_encoder.joblib`

---

## Run MoE Inference

```bash
python moe2.py
```

This will:

* Route user queries to the correct expert
* Load the associated LoRA adapter
* Produce specialist mental-health responses

---

## Important Disclaimer

This system deals with **mental-health–related content**.
It is **not a clinical or diagnostic tool**. Outputs must be reviewed by a qualified human professional.

---

## License

Internal academic/testing use only unless specified otherwise.
