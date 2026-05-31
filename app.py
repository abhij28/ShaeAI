import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShaeAI Skin Analyzer",
    page_icon="🧴",
    layout="centered"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "face_skin_best.pth"
ENCODER_PATH = "label_encoder.pkl"

# ── Model ─────────────────────────────────────────────────────────────────────
class SkinConditionModel(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features   = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ── Load model (cached so loads only once) ────────────────────────────────────
@st.cache_resource
def load_model():
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SkinConditionModel(out_classes=len(encoder.classes_)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, encoder, device

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Predict function ──────────────────────────────────────────────────────────
def predict(image, model, encoder, device):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs    = model(tensor)
        probs      = torch.softmax(outputs, dim=1)[0]
        confidence = probs.max().item() * 100
        pred_idx   = probs.argmax().item()
    condition  = encoder.inverse_transform([pred_idx])[0]
    all_probs  = {
        encoder.inverse_transform([i])[0]: round(probs[i].item() * 100, 1)
        for i in range(len(encoder.classes_))
    }
    return condition, round(confidence, 1), all_probs

# ── Skin tips ─────────────────────────────────────────────────────────────────
TIPS = {
    "Pimples / Acne"               : "🧴 Use salicylic acid cleanser. Avoid touching face. Keep pillowcase clean.",
    "Redness / Rosacea"            : "❄️ Use gentle fragrance-free products. Avoid hot water and spicy food.",
    "Dryness / Dehydrated Skin"    : "💧 Use hyaluronic acid serum. Moisturize immediately after washing.",
    "Hyperpigmentation / Black Spots": "☀️ Use SPF 50 daily. Try niacinamide or vitamin C serum.",
    "Blackheads / Open Pores"      : "🫧 Use BHA exfoliant. Try clay mask weekly. Don't squeeze.",
    "Whiteheads"                   : "✨ Use gentle retinol. Keep skin clean. Avoid heavy creams.",
    "Under-eye Bags"               : "😴 Get enough sleep. Use cold compress. Try caffeine eye cream.",
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧴 ShaeAI Skin Analyzer")
st.markdown("Analyze your skin condition using AI — upload a photo or use your camera.")
st.markdown("---")

# Load model
with st.spinner("Loading AI model..."):
    model, encoder, device = load_model()
st.success(f"✅ Model ready — {len(encoder.classes_)} skin conditions")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Live Camera"])

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload a face photo")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear front-facing photo"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Skin", key="upload_btn"):
            with st.spinner("Analyzing..."):
                condition, confidence, all_probs = predict(image, model, encoder, device)

            st.markdown("---")
            st.markdown(f"### Result: **{condition}**")
            st.markdown(f"**Confidence:** {confidence}%")
            st.progress(int(confidence))

            if condition in TIPS:
                st.info(f"**Skin Tip:** {TIPS[condition]}")

            with st.expander("📊 See all probabilities"):
                sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
                for label, prob in sorted_probs.items():
                    st.write(f"{label}: {prob}%")
                    st.progress(int(prob))

# ── Tab 2: Camera ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Take a photo with your camera")
    camera_photo = st.camera_input("Point camera at your face and click")

    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")

        if st.button("🔍 Analyze Skin", key="camera_btn"):
            with st.spinner("Analyzing..."):
                condition, confidence, all_probs = predict(image, model, encoder, device)

            st.markdown("---")
            st.markdown(f"### Result: **{condition}**")
            st.markdown(f"**Confidence:** {confidence}%")
            st.progress(int(confidence))

            if condition in TIPS:
                st.info(f"**Skin Tip:** {TIPS[condition]}")

            with st.expander("📊 See all probabilities"):
                sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
                for label, prob in sorted_probs.items():
                    st.write(f"{label}: {prob}%")
                    st.progress(int(prob))

st.markdown("---")
st.caption("ShaeAI — AI Powered Skin Analysis")
