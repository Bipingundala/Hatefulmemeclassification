import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import gdown

# Google Drive file ID of your model
MODEL_FILE_ID = "https://drive.google.com/file/d/1vtfq0zb_Kb2Mf2rW9P_phnAVUuEMbOlD/view?usp=sharing"
MODEL_FILE_NAME = "hateclipper_model_finetuned1.pt"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_FILE_NAME):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        st.info("Downloading model from Google Drive. This may take a while...")
        gdown.download(url, MODEL_FILE_NAME, quiet=False)
    else:
        st.success("Model already downloaded.")
    return MODEL_FILE_NAME

@st.cache_resource
def load_model(model_path):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

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
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model, processor

# Download model once
model_path = download_model()

# Load model and processor
model, processor = load_model(model_path)

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
