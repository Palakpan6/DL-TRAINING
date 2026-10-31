from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf


MODEL_PATH = Path(__file__).resolve().parent / "mnist_generator.keras"
LATENT_DIM = 100

st.title("Digit Image Generator")
st.write("Choose a digit from 0 to 9 and generate an image.")

digit = st.number_input("Digit", min_value=0, max_value=9, value=0, step=1)

@st.cache_resource
def load_generator():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Generator model not found at {MODEL_PATH}. "
            "Train and save a conditional generator as mnist_generator.keras "
            "in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

if st.button("Generate image"):
    try:
        generator = load_generator()

        noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
        labels = np.array([digit], dtype=np.int32)
        image = generator.predict([noise, labels], verbose=0)[0]

        # Convert common generator output ranges to displayable pixels.
        image = np.squeeze(image)
        if image.min() < 0:
            image = (image + 1) / 2
        image = np.clip(image, 0, 1)

        st.image(image, caption=f"Generated digit: {digit}", clamp=True)
    except FileNotFoundError:
        st.error(
            f"{MODEL_PATH} is missing. Run `python train_generator.py` from the "
            "project folder to train and save the conditional generator."
        )
    except Exception as error:
        st.error(f"Could not generate an image. Check the generator's input format. Details: {error}")
