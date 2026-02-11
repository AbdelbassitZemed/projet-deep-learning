import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(
    page_title="Détection des maladies des plantes",
    layout="centered"
)

st.title("🌿 Détection des maladies des plantes")
st.write("Application basée sur YOLO11 fine-tuné sur le dataset PlantDoc")

# -----------------------------
# Chargement du modèle
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("runs_finetune/yolo11s_ft_cpu/weights/best.pt")

model = load_model()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Télécharger une image de feuille",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("📷 Image d'entrée")
    st.image(image, width=450)

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # -----------------------------
    # Détection
    # -----------------------------
    st.subheader("🔍 Détection en cours...")
    results = model.predict(source=image_path, imgsz=512, conf=0.25)

    # -----------------------------
    # Résultat
    # -----------------------------
    st.subheader("✅ Résultat de la détection")
    result_image = results[0].plot()
    st.image(result_image, width=450)

    os.remove(image_path)
