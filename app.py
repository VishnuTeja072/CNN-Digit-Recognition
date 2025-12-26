import streamlit as st
from PIL import Image
from src.predict import predict_digit

st.set_page_config(page_title="CNN Digit Recognition", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9).")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=150)

    if st.button("Predict"):
        digit = predict_digit(image)
        st.success(f"🧠 Predicted Digit: **{digit}**")
