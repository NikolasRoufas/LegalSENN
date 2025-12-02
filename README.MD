
# **LegalSENN: Self-Explaining Neural Networks for Legal Text Classification**

> **Property of**  
> **Nikolaos Roufas**  
> **Sapienza Università di Roma**  
> **All rights reserved.**  
> This codebase is proprietary and may not be redistributed or used without explicit permission from the author and Sapienza University.

---

## **Overview**

**LegalSENN** is an interpretable neural architecture for legal text classification.  
It combines:

- a **pretrained Transformer encoder** (e.g., *Legal-BERT*), and  
- a **Self-Explaining Neural Network (SENN) head**

to produce **both predictions and explanations** for legal documents.

The primary use case is training on **LexGLUE** tasks (e.g., *SCOTUS*, *EURLEX*) and generating:

- **jurisdiction / issue-area predictions**
- **concept-based explanations**  
  for a user-uploaded trial or judgment.

---

## **Core Ideas**

Given an input document \(x\), LegalSENN computes:

1. **Document representation**  
   \( z(x) \) from a pretrained encoder (e.g., Legal-BERT).

2. **Concept activations**  
   \( h_i(x) \) via a linear *concept encoder*.

3. **Relevance scores**  
   \( \alpha_i(x) \) via an MLP *relevance predictor*.

4. **Concept contributions**  
   \[
   c_i(x) = h_i(x) \cdot \alpha_i(x)
   \]

5. **Final classifier**  
   \[
   f(x) = W_g \, c(x) + b_g
   \]

This architecture follows **Self-Explaining Neural Networks (SENNs)**  
by *Alvarez-Melis & Jaakkola (2018)*, adapted to the domain of legal text classification.

---

## **Repository Structure**

Flat project layout:

```

Self-Explainable-Neural-Networks/
│
├─ model.py                  # LegalSENN architecture (encoder + SENN head)
├─ data.py                   # Dataset + label map utilities
├─ train.py                  # Training loop for LexGLUE CSVs
├─ inference.py              # Inference + explanation on a single document
├─ prepare_lexglue_csv.py    # Convert LexGLUE subsets to text,label CSVs
├─ requirements.txt          # Python dependencies
│
├─ data/
│   └─ lexglue_csv/          # Generated CSVs (train/val/test per task)
│
├─ outputs/
│   └─ legal_senn_*          # Saved models, configs, label maps
│
└─ venv/                     # Python virtual environment (ignored by git)

````

---

## **Setup**

### 1. Create and activate virtual environment (macOS / Linux)

```bash
python3 -m venv venv
source venv/bin/activate
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

* `torch`
* `transformers`
* `datasets`
* `pandas`
* `tqdm`

On Apple Silicon (M1/M2/M3), PyTorch will automatically use **MPS** if available.

---

## **Dataset: LexGLUE (SCOTUS Example)**

This project uses **LexGLUE** via Hugging Face Datasets:
[https://github.com/coastalcph/lex-glue](https://github.com/coastalcph/lex-glue)

Prepare CSVs for a specific task (e.g., *SCOTUS*):

```bash
python prepare_lexglue_csv.py \
  --task scotus \
  --output_dir data/lexglue_csv \
  --primary_label_only
```

Generates:

```
data/lexglue_csv/scotus/train.csv
data/lexglue_csv/scotus/val.csv
data/lexglue_csv/scotus/test.csv
```

Format:

```
text,label
"<full legal document text>", "0"
"<another document>", "5"
...
```

For multi-label tasks (*e.g., EURLEX*), only the **first (primary) label** is used for compatibility with `CrossEntropyLoss`.

---

## **Training LegalSENN**

Example: Train on **SCOTUS** using **Legal-BERT**.

```bash
python train.py \
  --train_csv data/lexglue_csv/scotus/train.csv \
  --val_csv   data/lexglue_csv/scotus/val.csv \
  --encoder_name nlpaueb/legal-bert-base-uncased \
  --output_dir outputs/legal_senn_scotus \
  --num_epochs 3 \
  --batch_size 2 \
  --max_length 512 \
  --num_concepts 16
```

During training, you will see:

* Training loss / accuracy per epoch
* Validation loss / accuracy per epoch

Best model saved as:

```
outputs/legal_senn_scotus/best_model.pt
outputs/legal_senn_scotus/config.json
outputs/legal_senn_scotus/label_map.json
```

### **MPS fallback (optional)**

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## **Inference: Predicting on a User Trial**

Prepare a text file:

```bash
nano my_trial.txt
```

Run inference:

```bash
python inference.py \
  --model_dir outputs/legal_senn_scotus \
  --text_file my_trial.txt \
  --top_k 3
```

Output example:

```
=== Top-k predicted labels ===
3                         prob=0.7213
5                         prob=0.1832
1                         prob=0.0955

=== Concept contributions (SENN head) ===
Concept 03: contribution= 2.1145
Concept 07: contribution= 1.8733
Concept 01: contribution=-0.4542
...
```

Alternatively, pass raw text:

```bash
python inference.py \
  --model_dir outputs/legal_senn_scotus \
  --text "Full text of the trial goes here..." \
  --top_k 3
```

---


## **Ownership and License**

This codebase is:

* **Property of**: *Nikolaos Roufas* and *Sapienza Università di Roma*
* **Status**: *Proprietary (all rights reserved)*

**this code must not be redistributed, modified, or used commercially**
without prior written permission.
