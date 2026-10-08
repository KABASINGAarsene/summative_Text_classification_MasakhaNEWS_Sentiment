import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Hugging Face Model Repository ID
MODEL_PATH = "LJMJ/afriberta-masakhanews"

# Fallback label mapping matching training order
DEFAULT_ID2LABEL = {
    0: "business",
    1: "entertainment",
    2: "health",
    3: "politics",
    4: "religion",
    5: "sports",
    6: "technology"
}

# Load model, tokenizer, and labels once into server memory
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Extract labels directly from the model configuration
    if hasattr(model.config, "id2label") and model.config.id2label:
        label_mapping = {int(k): v for k, v in model.config.id2label.items()}
    else:
        label_mapping = DEFAULT_ID2LABEL
        
    return tokenizer, model, label_mapping

tokenizer, model, label_mapping = load_model()

# Page Configuration & Header
st.set_page_config(page_title="AfriNews AI", page_icon="📰", layout="centered")

st.title("📰 AfriNews AI: Multilingual Topic Classifier")
st.markdown(
    "Classify news articles in **Hausa** and **Swahili** across 7 editorial domains: "
    "*Business, Entertainment, Health, Politics, Religion, Sports, and Technology*."
)

# Sidebar with model metadata and quick test samples
with st.sidebar:
    st.header("Model Details")
    st.markdown("**Architecture:** Fine-Tuned AfriBERTa (`afriberta_small`)")
    st.markdown("**Benchmark:** MasakhaNEWS Corpus")
    st.markdown("**Macro F1-Score:** 87.59% | **Accuracy:** 88.68%")
    st.markdown("---")
    st.subheader("Quick Test Samples")
    st.caption("Click any button to paste a real headline into the input area:")
    
    sample_hausa_sports = "Kocin Manchester United ya bayyana dalilin da ya sa kungiyar ta sha kaye a wasan karshe."
    sample_hausa_politics = "Muhammadu Buhari: PDP ta ce gwamnatin APC ta jawo kunci da talauci a Najeriya."
    sample_swahili_health = "Wizara ya Afya yatangaza mikakati mipya ya kupambana na ugonjwa wa malaria nchini."
    sample_swahili_tech = "Kampuni ya teknolojia yazindua simu mpya ya kisasa inayotumia mtandao wa 5G."

    if st.button("Hausa (Sports)"):
        st.session_state["input_text"] = sample_hausa_sports
    if st.button("Hausa (Politics)"):
        st.session_state["input_text"] = sample_hausa_politics
    if st.button("Swahili (Health)"):
        st.session_state["input_text"] = sample_swahili_health
    if st.button("Swahili (Technology)"):
        st.session_state["input_text"] = sample_swahili_tech

# Input text box
default_val = st.session_state.get("input_text", "")
user_input = st.text_area("Paste your news headline or article excerpt here:", value=default_val, height=140)

# Prediction execution
if st.button("Classify Article", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing text..."):
            # Tokenize input
            inputs = tokenizer(
                user_input, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            )
            
            # Forward pass inference
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Calibrated class probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
            
            # Map predictions to class labels
            prob_dict = {label_mapping.get(i, f"Class {i}"): prob for i, prob in enumerate(probabilities)}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            top_label, top_prob = sorted_probs[0]
            
            # Display primary prediction
            st.success(f"**Predicted Category:** {top_label.upper()} (Confidence: {top_prob*100:.1f}%)")
            
            st.divider()
            
            # Display full probability distribution
            st.write("### Category Probability Distribution:")
            for label, prob in sorted_probs:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**{label.capitalize()}**")
                with col2:
                    st.progress(float(prob), text=f"{prob*100:.1f}%")
    else:
        st.warning("Please enter or select a headline to classify.")