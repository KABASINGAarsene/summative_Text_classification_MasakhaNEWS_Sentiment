#  AfriNews AI — Multilingual African News Topic Classifier

A fine-tuned **AfriBERTa** (`afriberta_small`) model for classifying African news articles in **Hausa** and **Swahili** into 7 editorial categories, trained on the **MasakhaNEWS** corpus.

>  **Model on Hugging Face:** [LJMJ/afriberta-masakhanews](https://huggingface.co/LJMJ/afriberta-masakhanews)

---

##  Overview

| Property         | Detail                                             |
|------------------|----------------------------------------------------|
| **Task**         | Multi-class text classification                    |
| **Languages**    | Hausa (`hau`), Swahili (`swa`)                     |
| **Categories**   | Business, Entertainment, Health, Politics, Religion, Sports, Technology |
| **Base Model**   | AfriBERTa (`afriberta_small`) — `xlm-roberta` arch |
| **Architecture** | `XLMRobertaForSequenceClassification`              |
| **Dataset**      | [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news) |
| **License**      | Apache 2.0                                         |

---

##  Performance

| Metric         | Score    |
|----------------|----------|
| **Macro F1**   | 87.59%   |
| **Accuracy**   | 88.68%   |

---

##  Model Architecture

```
Architecture    : XLMRobertaForSequenceClassification
Hidden size     : 768
Attention heads : 6
Hidden layers   : 4
Max tokens      : 514
Vocab size      : 70,006
Problem type    : single_label_classification
```

---

##  Dataset — MasakhaNEWS

MasakhaNEWS is a benchmark corpus for African-language news topic classification spanning multiple low-resource African languages. This project focuses on:

- **Hausa** — widely spoken in West Africa (Nigeria, Niger, Ghana)
- **Swahili** — widely spoken in East Africa (Kenya, Tanzania, Uganda)

---

##  Quick Start

### Prerequisites

```bash
pip install streamlit torch transformers
```

### Run the Streamlit App

```bash
streamlit run app.py
```

### Use the Model Directly (Python)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_ID = "LJMJ/afriberta-masakhanews"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

text = "Kampuni ya teknolojia yazindua simu mpya ya kisasa inayotumia mtandao wa 5G."

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = logits.argmax(-1).item()
print(model.config.id2label[predicted_class])  # → 'technology'
```

---

##  Label Mapping

| ID | Category      |
|----|---------------|
| 0  | Business      |
| 1  | Entertainment |
| 2  | Health        |
| 3  | Politics      |
| 4  | Religion      |
| 5  | Sports        |
| 6  | Technology    |

---

##  Project Structure

```
summative_Text_classification_MasakhaNEWS_Sentiment/
├── app.py                                          # Streamlit web app
├── requirements.txt                                # Python dependencies
├── saved_afriberta_news_model/                     # Local model checkpoint
└── text-classification-masakhanews-sentiment.ipynb # Training notebook
```

---

##  Streamlit App Features

- **Paste** any Hausa or Swahili headline / article excerpt
- **One-click samples** for quick testing (Sports, Politics, Health, Technology)
- **Confidence score** for the top predicted category
- **Full probability distribution** shown as a progress bar chart
- **Automatic language detection** (Hausa vs. Swahili)

---

##  Dependencies

```
streamlit
torch
transformers
```

---

##  References

- [MasakhaNEWS Dataset](https://github.com/masakhane-io/masakhane-news)
- [AfriBERTa Paper](https://arxiv.org/abs/2111.11921)
- [Hugging Face Model Hub — LJMJ/afriberta-masakhanews](https://huggingface.co/LJMJ/afriberta-masakhanews)
