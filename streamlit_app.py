import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle  # or use joblib if preferred
st.set_page_config(
    page_title="AI vs Human Art Classifier",
    layout="centered"
)

# ✅ Load the real classifier
@st.cache_resource
def load_model():
    with open("/Users/vivian_l/git/polygence_project/models_art_classifier.pkl", "rb") as file:
        loaded_data = pickle.load(file)

    model_config = loaded_data["config"]
    model_weights = loaded_data["weights"]

    model = tf.keras.models.model_from_json(model_config)
    model.set_weights(model_weights)
    return model



# Preprocess image depending on what your model expects
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # normalize
    if len(image_array.shape) == 3:  # If RGB, reshape to (1, H, W, C)
        image_array = image_array.reshape(1, 224, 224, 3)
    else:  # If flattened model
        image_array = image_array.flatten().reshape(1, -1)
    return image_array

# 🔮 Real prediction function
def predict(image_array, model):
    prob = model.predict(image_array)[0][0]  # Get scalar value
    label = "Human-Made Art" if prob >= 0.5 else "AI-Generated Art"
    return label, prob


# Streamlit UI below (same as yours)
# ... CSS styling (same as before) ...
# ... UI layout & upload (same as before) ...

st.title("🎨 AI vs Human Art Classifier")
st.markdown("<p style='font-size:18px;'>Upload an image and let the classifier decide if it's AI-generated or human-made.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)


    model = load_model()
    st.write("✅ Model loaded!")
    processed_image = preprocess_image(image)
    label, confidence = predict(processed_image, model)

    if label == "AI-Generated Art":
        text_color = "#bf0a30"
        emoji = "🤖"
        box_color = "#ffe6ee"
        border_color = "#ffe6ee"
    else:
        text_color = "#52cc02"
        emoji = "🎨"
        box_color = "#d4ecd6"
        border_color = "#d4ecd6"

    st.markdown(
        f"""
        <div style="
            border: 2px solid {border_color};
            background-color: {box_color};
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
        ">
            <h3 style='margin-bottom: 10px; color: {text_color};'>
                {emoji} Prediction: <strong>{label}</strong>
            </h3>
            <p style='font-size: 18px; color: {text_color};'>
                🔍 Confidence: <strong>{confidence:.2f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True
    )

else:
    st.markdown("<p style='font-size:18px;'>Please upload an image.</p>", unsafe_allow_html=True)
