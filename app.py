import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

# Defining the path to the extracted model folder
MODEL_PATH = "LJMJ/afriberta-masakhanews"

# Loading model, tokenizer, and labels once into memory
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    with open(f"{MODEL_PATH}/label_mapping.json", "r") as f:
        label_mapping = json.load(f)
        
    return tokenizer, model, label_mapping

tokenizer, model, label_mapping = load_model()

# User Interface Layout
st.set_page_config(page_title="AfriNews AI", page_icon="📰")

st.title("📰 AfriNews AI: Multilingual Topic Classifier")
st.markdown("Classify news articles in **Hausa** and **Swahili** across 7 editorial domains: *Business, Entertainment, Health, Politics, Religion, Sports, and Technology*.")

# Text input box
user_input = st.text_area("Paste your news headline or article excerpt here:", height=150)

# Prediction execution
if st.button("Classify Article", type="primary"):
    if user_input.strip():
        # Tokenize the input text
        inputs = tokenizer(
            user_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        )
        
        # Getting model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Converting logits to probabilities using Softmax
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
        
        # Mapping probabilities to category names and sorting them
        prob_dict = {label_mapping[str(i)]: prob for i, prob in enumerate(probabilities)}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Extracting the winning category
        top_label = sorted_probs[0][0]
        top_prob = sorted_probs[0][1]
        
        # Displaying the main result
        st.success(f"**Prediction:** {top_label.upper()} (Confidence: {top_prob*100:.1f}%)")
        
        st.divider()
        
        # Displaying the distribution of probabilities
        st.write("### Class Probabilities:")
        for label, prob in sorted_probs:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"**{label.capitalize()}**")
            with col2:
                # Rendering a visual progress bar for each category
                st.progress(float(prob), text=f"{prob*100:.1f}%")
    else:
        st.warning("Please enter some text to classify.")