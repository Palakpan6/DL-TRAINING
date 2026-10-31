import os
import shutil
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from CNN_MODEL import build_cnn
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="📸",
    layout="wide"
)

st.title("🧠 CNN Image Classification Studio")
st.caption("Train Convolutional Neural Networks on CIFAR-10 or custom datasets, inspect performance curves, and run live image predictions.")

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "img_size" not in st.session_state:
    st.session_state.img_size = 32
if "class_names" not in st.session_state:
    st.session_state.class_names = []
if "train_history" not in st.session_state:
    st.session_state.train_history = None

# Standard CIFAR-10 class labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ---------------------------------------------------------
# Custom Keras Callback for Streamlit Progress
# ---------------------------------------------------------
class StreamlitTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, progress_bar, status_box):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_box = status_box

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_box.markdown(
            f"**Epoch {epoch + 1}/{self.total_epochs}** — "
            f"Train Acc: `{acc:.4f}` | Val Acc: `{val_acc:.4f}` | "
            f"Train Loss: `{loss:.4f}` | Val Loss: `{val_loss:.4f}`"
        )


# ---------------------------------------------------------
# Helper: Find Image Dataset Root in Extracted ZIP
# ---------------------------------------------------------
def find_dataset_root(extracted_path):
    """
    Finds the directory that contains the class subfolders (e.g., cat/, dog/).
    Handles nested zip folders and ignores __MACOSX.
    """
    # Remove macOS metadata if present
    macosx_dir = os.path.join(extracted_path, "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir, ignore_errors=True)

    # Check current directory
    subdirs = [d for d in os.listdir(extracted_path) if os.path.isdir(os.path.join(extracted_path, d))]

    # If this directory has subdirectories with image files, it's the class root
    for subdir in subdirs:
        subdir_path = os.path.join(extracted_path, subdir)
        files = os.listdir(subdir_path)
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            return extracted_path

    # If there is a single wrapper directory, dive into it
    if len(subdirs) == 1:
        nested = os.path.join(extracted_path, subdirs[0])
        return find_dataset_root(nested)

    return extracted_path


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.header("⚙️ Navigation")
section = st.sidebar.radio("Go to:", ["Train Model", "Predict with Model"])

# Model Status Card in Sidebar
st.sidebar.divider()
st.sidebar.subheader("📦 Model Status")
if st.session_state.model is not None:
    st.sidebar.success("✅ Model Trained & Loaded")
    st.sidebar.write(f"**Input Size:** `{st.session_state.img_size}x{st.session_state.img_size}x3`")
    st.sidebar.write(f"**Classes ({len(st.session_state.class_names)}):** {', '.join(st.session_state.class_names[:5])}{'...' if len(st.session_state.class_names) > 5 else ''}")
else:
    st.sidebar.info("ℹ️ No model loaded in memory. Train a model or upload a saved `.keras` file.")


# ---------------------------------------------------------
# SECTION 1: TRAIN MODEL
# ---------------------------------------------------------
if section == "Train Model":
    st.subheader("🏋️ Model Training Setup")

    col_dataset, col_hyper = st.columns([1, 1])

    with col_dataset:
        st.markdown("### 1. Select Dataset")
        dataset_choice = st.radio(
            "Choose data source:",
            ["CIFAR-10 (Built-In 10 Classes)", "Custom Dataset (ZIP Upload)"],
            index=0
        )

    with col_hyper:
        st.markdown("### 2. Hyperparameters")
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        epochs = st.slider("Number of Epochs", min_value=1, max_value=20, value=5)

    st.divider()

    # ---------------------------------------------------------
    # CASE 1: CIFAR-10 Dataset
    # ---------------------------------------------------------
    if dataset_choice == "CIFAR-10 (Built-In 10 Classes)":
        st.markdown("### 📦 CIFAR-10 Configuration")
        st.info("The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*.")

        sample_subset = st.select_slider(
            "Select Training Set Size (Smaller size = faster CPU training):",
            options=[2000, 5000, 10000, 20000, 50000],
            value=5000,
            help="Choose 5,000 for a quick 30-second demo on CPU, or 50,000 for full training."
        )

        test_subset = min(int(sample_subset * 0.2), 10000)

        if st.button("🚀 Train CNN on CIFAR-10", type="primary", use_container_width=True):
            with st.spinner("Loading CIFAR-10 dataset..."):
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()

                # Slice subset for speed
                x_train = x_train[:sample_subset] / 255.0
                y_train = y_train[:sample_subset]
                x_test = x_test[:test_subset] / 255.0
                y_test = y_test[:test_subset]

                y_train_cat = to_categorical(y_train, 10)
                y_test_cat = to_categorical(y_test, 10)

            st.write(f"Training on **{x_train.shape[0]:,} samples**, validating on **{x_test.shape[0]:,} samples**...")

            # Build model
            model = build_cnn(input_shape=(32, 32, 3), num_classes=10)

            # Progress UI
            progress_bar = st.progress(0.0)
            status_box = st.empty()
            cb = StreamlitTrainingCallback(epochs, progress_bar, status_box)

            with st.spinner("Training model in progress..."):
                history = model.fit(
                    x_train, y_train_cat,
                    validation_data=(x_test, y_test_cat),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[cb],
                    verbose=0
                )

            # Store in session state
            st.session_state.model = model
            st.session_state.img_size = 32
            st.session_state.class_names = CIFAR10_CLASSES
            st.session_state.train_history = history.history

            st.success("🎉 Model training successfully completed!")

    # ---------------------------------------------------------
    # CASE 2: Custom ZIP Dataset Upload
    # ---------------------------------------------------------
    else:
        st.markdown("### 📁 Custom Dataset Upload")
        st.write("Upload a `.zip` file containing subfolders organized by class (e.g., `cats/` and `dogs/`).")

        img_size = st.slider("Target Image Size (px)", min_value=32, max_value=256, value=64, step=16)
        zip_file = st.file_uploader("Upload dataset ZIP archive", type=["zip"])

        if zip_file:
            temp_extract_dir = tempfile.mkdtemp(prefix="cnn_data_")
            try:
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(temp_extract_dir)

                data_root = find_dataset_root(temp_extract_dir)
                train_dir = os.path.join(data_root, "train")
                test_dir = os.path.join(data_root, "test")
                train_classes = sorted(
                    d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))
                ) if os.path.isdir(train_dir) else []
                test_classes = sorted(
                    d for d in os.listdir(test_dir)
                    if os.path.isdir(os.path.join(test_dir, d))
                ) if os.path.isdir(test_dir) else []
                has_train_test_split = bool(train_classes) and train_classes == test_classes
                classes_found = train_classes if has_train_test_split else [
                    d for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))
                ]

                st.success(f"✅ Found **{len(classes_found)} classes**: `{', '.join(classes_found)}`")

                if has_train_test_split:
                    st.info("Using 20% of `train/` for validation; the separate `test/` set is held out for final evaluation.")
                    train_datagen = ImageDataGenerator(
                        rescale=1.0 / 255.0,
                        validation_split=0.2
                    )
                    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
                    train_gen = train_datagen.flow_from_directory(
                        train_dir,
                        target_size=(img_size, img_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset='training',
                        shuffle=True
                    )
                    val_gen = train_datagen.flow_from_directory(
                        train_dir,
                        target_size=(img_size, img_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset='validation',
                        shuffle=False
                    )
                    test_gen = test_datagen.flow_from_directory(
                        test_dir,
                        target_size=(img_size, img_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False
                    )
                else:
                    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
                    train_gen = datagen.flow_from_directory(
                        data_root,
                        target_size=(img_size, img_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset='training',
                        shuffle=True
                    )
                    val_gen = datagen.flow_from_directory(
                        data_root,
                        target_size=(img_size, img_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset='validation',
                        shuffle=False
                    )

                st.write(f"📊 Total Training Images: **{train_gen.samples}**, Validation Images: **{val_gen.samples}**")

                if st.button("🚀 Train CNN on Uploaded Dataset", type="primary", use_container_width=True):
                    model = build_cnn(input_shape=(img_size, img_size, 3), num_classes=train_gen.num_classes)

                    progress_bar = st.progress(0.0)
                    status_box = st.empty()
                    cb = StreamlitTrainingCallback(epochs, progress_bar, status_box)

                    with st.spinner("Training on uploaded images..."):
                        history = model.fit(
                            train_gen,
                            validation_data=val_gen,
                            epochs=epochs,
                            callbacks=[cb],
                            verbose=0
                        )

                    if has_train_test_split:
                        test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)

                    st.session_state.model = model
                    st.session_state.img_size = img_size
                    # Map sorted indices to class names
                    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
                    st.session_state.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
                    st.session_state.train_history = history.history

                    st.success("🎉 Custom dataset training finished successfully!")
                    if has_train_test_split:
                        st.metric("Held-out Test Accuracy", f"{test_accuracy:.2%}")
                        st.caption(f"Test loss: {test_loss:.4f} on {test_gen.samples} held-out images.")

            except Exception as e:
                st.error(f"Error reading dataset ZIP: {e}")

    # ---------------------------------------------------------
    # Render Training Plots & Save Model
    # ---------------------------------------------------------
    if st.session_state.train_history is not None:
        st.divider()
        st.subheader("📈 Training & Validation Metrics")

        hist = st.session_state.train_history
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy Plot
        ax[0].plot(hist.get("accuracy", []), marker='o', label="Train Accuracy", color="#2563EB")
        if "val_accuracy" in hist:
            ax[0].plot(hist.get("val_accuracy", []), marker='s', label="Val Accuracy", color="#10B981")
        ax[0].set_title("Model Accuracy across Epochs")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Accuracy")
        ax[0].grid(True, linestyle="--", alpha=0.5)
        ax[0].legend()

        # Loss Plot
        ax[1].plot(hist.get("loss", []), marker='o', label="Train Loss", color="#DC2626")
        if "val_loss" in hist:
            ax[1].plot(hist.get("val_loss", []), marker='s', label="Val Loss", color="#F59E0B")
        ax[1].set_title("Model Loss across Epochs")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].grid(True, linestyle="--", alpha=0.5)
        ax[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Save model option
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            save_name = st.text_input("Model filename to save:", value="trained_cnn_model.keras")
        with save_col2:
            st.write("")
            st.write("")
            if st.button("💾 Save Model to Disk", use_container_width=True):
                st.session_state.model.save(save_name)
                st.success(f"Model saved as `{save_name}`!")


# ---------------------------------------------------------
# SECTION 2: PREDICT WITH MODEL
# ---------------------------------------------------------
elif section == "Predict with Model":
    st.subheader("🔍 Classify New Images")

    # Option to load a previously saved model if not in memory
    if st.session_state.model is None:
        st.warning("⚠️ No trained model is currently held in session memory.")
        st.markdown("You can either train a new model in the **Train Model** tab, or load a saved `.keras` / `.h5` file below:")

        saved_file = st.file_uploader("Upload Saved Model (.keras or .h5)", type=["keras", "h5"])
        load_size = st.number_input("Input Image Size for loaded model (px)", min_value=32, max_value=256, value=32, step=16)
        classes_str = st.text_input("Class names (comma-separated)", value=", ".join(CIFAR10_CLASSES))

        if saved_file and st.button("📥 Load Model", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(saved_file.name)[1]) as tmp:
                tmp.write(saved_file.getvalue())
                tmp_path = tmp.name
            try:
                loaded = load_model(tmp_path)
                st.session_state.model = loaded
                st.session_state.img_size = int(load_size)
                st.session_state.class_names = [c.strip() for c in classes_str.split(",") if c.strip()]
                st.success("✅ Model loaded successfully!")
                st.rerun()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # Prediction interface
    if st.session_state.model is not None:
        st.write(f"Ready to classify images with input resolution: **`{st.session_state.img_size}x{st.session_state.img_size}`**")

        uploaded_img = st.file_uploader("Choose an image to classify...", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_img:
            col_img, col_pred = st.columns([1, 1])

            # Prepare image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_img.getvalue())
                tmp_img_path = tmp.name

            try:
                img_display = load_img(tmp_img_path)
                img_target = load_img(tmp_img_path, target_size=(st.session_state.img_size, st.session_state.img_size))

                with col_img:
                    st.image(img_display, caption="Uploaded Image", use_container_width=True)

                # Preprocess for model
                img_array = img_to_array(img_target) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Inference
                preds = st.session_state.model.predict(img_array)[0]
                top_idx = int(np.argmax(preds))
                top_conf = float(preds[top_idx]) * 100.0

                class_label = st.session_state.class_names[top_idx] if top_idx < len(st.session_state.class_names) else f"Class {top_idx}"

                with col_pred:
                    st.markdown("### Prediction Result")
                    st.success(f"🎯 **Predicted Class:** `{class_label.upper()}`")
                    st.metric(label="Confidence", value=f"{top_conf:.2f}%")

                    # Confidence distribution bar chart
                    st.markdown("#### Probability Distribution")
                    labels = st.session_state.class_names if len(st.session_state.class_names) == len(preds) else [f"C{i}" for i in range(len(preds))]
                    chart_data = {label: float(p) for label, p in zip(labels, preds)}
                    st.bar_chart(chart_data)

            finally:
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)