"""
app.py — Gradio-based inference app for the AfriCAT/MasakhaNEWS
         African news category classifier.

Model:  saved_afriberta_news_model  (XLM-RoBERTa fine-tuned on MasakhaNEWS)
Labels: business | entertainment | health | politics | religion | sports | technology
Languages supported: Hausa (hau) · Swahili (swa)
"""

import json
import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_afriberta_news_model")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.json")

# ---------------------------------------------------------------------------
# Load model & tokenizer (done once at startup)
# ---------------------------------------------------------------------------
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print("Loading model …")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load label mapping  { "0": "business", "1": "entertainment", … }
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    id2label: dict[str, str] = json.load(f)

# Reverse mapping for quick lookup
label2id: dict[str, int] = {v: int(k) for k, v in id2label.items()}

# Ordered list of labels (sorted by integer id)
LABELS = [id2label[str(i)] for i in range(len(id2label))]

# Emoji decoration for each category
CATEGORY_EMOJI = {
    "business":      "💼",
    "entertainment": "🎭",
    "health":        "🏥",
    "politics":      "🏛️",
    "religion":      "⛪",
    "sports":        "⚽",
    "technology":    "💻",
}

MAX_LENGTH = 256   # tokens — matches training truncation


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------
def classify(headline: str, body: str) -> dict:
    """
    Combine headline + body (matching the training feature `full_text`),
    tokenise, run the model and return a {label: confidence} dict for Gradio.
    """
    headline = headline.strip()
    body = body.strip()

    if not headline and not body:
        return {lbl: 0.0 for lbl in LABELS}

    # Replicate the training feature construction: headline + " " + body
    full_text = f"{headline} {body}".strip()

    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    # Build the confidence dict Gradio Label component expects
    result = {}
    for idx, prob in enumerate(probs):
        label = id2label[str(idx)]
        emoji = CATEGORY_EMOJI.get(label, "")
        result[f"{emoji} {label}"] = float(prob)

    return result


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DESCRIPTION = """
## 🌍 AfriCAT — African News Category Classifier

Fine-tuned on the **[MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews)**
dataset (Hausa + Swahili) using **AfriBERTa** (XLM-RoBERTa architecture).

**Categories:** Business · Entertainment · Health · Politics · Religion · Sports · Technology

Enter a headline and/or article body in **Hausa** or **Swahili** and the model will predict
the most likely news category.
"""

EXAMPLES = [
    # Hausa examples
    [
        "Yadda Shugaban Kungiyar Kirista ya gina masallaci a Najeriya",
        "Shugaban wata kungiyar Kirista ta kasar Najeriya ya sanar da gina masallacin jama'a a yankin arewacin kasar.",
    ],
    [
        "Manchester United ta doke Chelsea 3-1 a gasar Premier League",
        "A wasan da aka buga a filin Old Trafford, Manchester United ta samu nasara mai kayatarwa akan Chelsea.",
    ],
    # Swahili examples
    [
        "Serikali yatangaza mpango mpya wa afya",
        "Serikali imetangaza mpango mpya wa bima ya afya utakaonufaisha wananchi wote.",
    ],
    [
        "Biashara ya teknolojia inakua haraka Afrika",
        "Sekta ya teknolojia barani Afrika inaendelea kukua kwa kasi kubwa na kampuni nyingi mpya zinazoundwa.",
    ],
]

with gr.Blocks(title="AfriCAT — African News Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            headline_box = gr.Textbox(
                label="📰 Headline (Hausa or Swahili)",
                placeholder="e.g. Manchester United ta doke Chelsea…",
                lines=2,
            )
            body_box = gr.Textbox(
                label="📄 Article Body (optional)",
                placeholder="Paste the article text here…",
                lines=8,
            )
            submit_btn = gr.Button("🔍 Classify", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Predicted Category",
                num_top_classes=7,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[headline_box, body_box],
        outputs=label_output,
        fn=classify,
        cache_examples=False,
        label="📚 Try an example",
    )

    submit_btn.click(
        fn=classify,
        inputs=[headline_box, body_box],
        outputs=label_output,
    )

    gr.Markdown(
        """
---
**Model details**
- Architecture: `XLMRobertaForSequenceClassification` (4 hidden layers, 768 hidden size)
- Tokenizer vocabulary: 70 006 tokens
- Max input length: 256 tokens
- Training data: MasakhaNEWS — Hausa (`hau`) + Swahili (`swa`)
- Classes (7): business · entertainment · health · politics · religion · sports · technology
        """
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
