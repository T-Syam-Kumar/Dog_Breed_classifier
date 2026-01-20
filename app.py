import streamlit as st
import torch
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Dog Breed Classifier 🐶",
    page_icon="🐕",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🐶 Dog Breed Classifier</h1>
    <p style="text-align:center;color:gray;">
    Upload a dog image and get top breed predictions with confidence
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )
    model.eval()
    return processor, model

processor, model = load_model()

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "📤 Upload a dog image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### 🔍 Prediction Result")

        with st.spinner("Analyzing image..."):
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

            # Top 5 predictions
            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)

            labels = [
                model.config.id2label[idx.item()]
                for idx in top_indices
            ]
            confidences = [p.item() * 100 for p in top_probs]

        st.success(f"**Predicted Breed:** {labels[0]}")
        st.info(f"**Confidence:** {confidences[0]:.2f}%")

    # ---------------- BAR CHART ----------------
    st.markdown("### 📊 Top 5 Breed Confidence Scores")

    df = pd.DataFrame({
        "Breed": labels,
        "Confidence (%)": confidences
    })

    st.bar_chart(
        df.set_index("Breed"),
        use_container_width=True
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Built with ❤️ using Streamlit, PyTorch & Transformers</p>",
    unsafe_allow_html=True
)
