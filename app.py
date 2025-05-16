import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# Load Processor & Model
@st.cache_resource
def load_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # HateCLIPper Architecture
    class HateCLIPper(nn.Module):
        def __init__(self, clip_model):
            super(HateCLIPper, self).__init__()
            self.clip = clip_model
            self.classifier = nn.Linear(self.clip.config.projection_dim * 2, 2)  # binary classification

        def forward(self, input_ids, pixel_values, attention_mask):
            outputs = self.clip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            combined = torch.cat([image_embeds, text_embeds], dim=1)
            logits = self.classifier(combined)
            return logits

    model = HateCLIPper(clip_model)
    model.load_state_dict(torch.load("hateclipper_model_finetuned1.pt", map_location=torch.device('cpu')))
    model.eval()

    return model, processor

model, processor = load_model()

# Streamlit UI
st.title("🖼️ Hateful Meme Detector 🚫")
st.markdown("Upload a meme image and enter its caption to check if it's **hateful or non-hateful**.")

uploaded_file = st.file_uploader("Upload Meme Image", type=['png', 'jpg', 'jpeg'])
caption = st.text_area("Enter Meme Caption")

if uploaded_file is not None and caption:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Meme", use_column_width=True)

    # Processing
    encoding = processor(caption, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
    pixel_values = processor(images=image, return_tensors='pt')['pixel_values']

    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'], 
                        pixel_values=pixel_values, 
                        attention_mask=encoding['attention_mask'])
        
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    # Display Result
    if pred == 1:
        st.error("⚠️ This meme is **Hateful**")
    else:
        st.success("✅ This meme is **Non-Hateful**")

    st.subheader("Prediction Probabilities")
    st.write(f"Non-Hateful: {probs[0][0].item()*100:.2f}%")
    st.write(f"Hateful: {probs[0][1].item()*100:.2f}%")
