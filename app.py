import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# ECOSORT - WASTE MANAGEMENT USING CNN
# =====================================================

st.set_page_config(
    page_title="EcoSort | Smart Waste Management",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "waste_classifier.keras"
LABELS_PATH = "class_names.txt"
IMG_SIZE = (224, 224)

DISPOSAL_TIPS = {
    "plastic": "Empty and rinse recyclable plastic where appropriate. Follow local recycling guidelines.",
    "wet_waste": "Place suitable food scraps in a wet-waste collection or composting bin.",
    "paper": "Keep paper clean and dry before recycling.",
    "glass": "Handle glass carefully and use the appropriate glass-recycling collection.",
    "metal": "Empty metal containers and recycle them where accepted.",
    "e-waste": "Take electronic items to an authorized e-waste collection centre."
}

CATEGORY_INFO = {
    "e-waste": ("🔌", "E-Waste", "Electronic devices and components"),
    "glass": ("🍾", "Glass", "Glass bottles and containers"),
    "metal": ("🥫", "Metal", "Metal cans and containers"),
    "paper": ("📄", "Paper", "Paper and paper products"),
    "plastic": ("🧴", "Plastic", "Plastic bottles and packaging"),
    "wet_waste": ("🍃", "Wet Waste", "Food scraps and organic waste")
}

# =====================================================
# CSS - STYLING ONLY, NO HTML CONTENT BLOCKS
# =====================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0faf3, #edf3ff);
    color: #183629;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #17452e;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #8cbea0;
    border-radius: 14px;
    padding: 12px;
}

.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dcebe0;
    border-radius: 14px;
    padding: 16px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(255,255,255,0.85);
    border-radius: 16px;
}

[data-testid="stSidebar"] {
    background-color: #f0f6f1;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if not os.path.exists(LABELS_PATH):
        return []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def pretty_label(label):
    return label.replace("_", " ").replace("-", " ").title()


# =====================================================
# HEADER
# =====================================================

st.title("♻️ EcoSort")

st.subheader("Smart Waste Classification Using CNN")

st.write(
    "A cleaner planet starts with one image. "
    "Upload a waste photograph to identify its category "
    "and discover the recommended disposal method."
)

st.success(
    "Six categories supported: Plastic, Wet Waste, Paper, "
    "Glass, Metal and E-Waste."
)

st.divider()


# =====================================================
# WASTE CATEGORY CARDS
# =====================================================

st.header("Explore Waste Categories")

columns = st.columns(3)

for index, (label, info) in enumerate(CATEGORY_INFO.items()):
    icon, title, description = info

    with columns[index % 3]:
        with st.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(description)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("♻️ EcoSort")
    st.write("AI-powered waste classification")

    st.divider()

    st.subheader("How it works")

    st.write("1. Upload a waste image.")
    st.write("2. Preview your image.")
    st.write("3. Click Analyse Image.")
    st.write("4. View the predicted category.")
    st.write("5. Read the disposal recommendation.")

    st.divider()

    st.caption(
        "Student project: Waste Management Using CNN"
    )


# =====================================================
# LOAD MODEL AND LABELS
# =====================================================

model = load_model()
class_names = load_class_names()

st.divider()
st.header("Analyse Your Waste")

st.write(
    "Upload a clear image containing one main waste item. "
    "The model will run only after you click Analyse Image."
)

if model is None:
    st.error(
        "Model not found. Keep waste_classifier.keras "
        "in the same directory as app.py."
    )

if len(class_names) != 6:
    st.error(
        "class_names.txt must contain six labels from "
        "the same training run as your CNN model."
    )


# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Choose a clear, well-lit image."
)

if uploaded_file is not None:

    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        preview_col, details_col = st.columns([1, 1])

        with preview_col:
            st.subheader("Image Preview")
            st.image(image, use_container_width=True)

        with details_col:
            st.subheader("Image Details")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Width:** {image.width} pixels")
            st.write(f"**Height:** {image.height} pixels")

            st.info(
                "Image uploaded successfully. "
                "Click the button below when you are ready."
            )

        analyse_clicked = st.button(
            "♻️ Analyse Image",
            type="primary",
            use_container_width=True
        )

        if analyse_clicked:

            if model is None or len(class_names) != 6:
                st.error(
                    "The model or class labels could not be loaded."
                )

            else:
                with st.spinner("Analysing your image..."):

                    # Convert to RGB and resize to model input size.
                    resized_image = image.resize(
                        IMG_SIZE,
                        Image.Resampling.LANCZOS
                    )

                    image_array = np.asarray(
                        resized_image,
                        dtype=np.float32
                    )

                    # Add exactly one batch dimension.
                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    # Validate the input shape.
                    expected_shape = tuple(model.input_shape[1:])
                    actual_shape = tuple(image_array.shape[1:])

                    if actual_shape != expected_shape:

                        st.error(
                            f"Input shape mismatch. "
                            f"Expected {expected_shape}, "
                            f"received {actual_shape}."
                        )

                    else:

                        # Run CNN prediction.
                        predictions = model.predict(
                            image_array,
                            verbose=0
                        )[0]

                        if len(predictions) != len(class_names):

                            st.error(
                                "The model output does not match "
                                "the labels in class_names.txt."
                            )

                        else:
                            best_index = int(
                                np.argmax(predictions)
                            )

                            best_label = class_names[best_index]

                            confidence = float(
                                predictions[best_index]
                            )

                            # Store the result for Streamlit reruns.
                            st.session_state["ecosort_result"] = {
                                "filename": uploaded_file.name,
                                "label": best_label,
                                "confidence": confidence,
                                "scores": predictions.tolist()
                            }

    except Exception as exc:
        st.error(f"Unable to process the image: {exc}")


# =====================================================
# DISPLAY CLASSIFICATION RESULTS
# =====================================================

result = st.session_state.get("ecosort_result")

if (
    uploaded_file is not None
    and result is not None
    and result["filename"] == uploaded_file.name
):

    label = result["label"]
    confidence = result["confidence"]

    icon, title, description = CATEGORY_INFO.get(
        label,
        ("♻️", pretty_label(label), "Predicted waste category")
    )

    st.divider()
    st.header("Classification Results")

    result_col, confidence_col = st.columns([1.2, 1])

    with result_col:
        with st.container(border=True):
            st.markdown(f"# {icon}")
            st.subheader(title)
            st.write(description)

    with confidence_col:
        st.metric(
            "Model Score",
            f"{confidence * 100:.2f}%"
        )

        st.progress(
            max(0.0, min(1.0, confidence))
        )

        st.caption(
            "The model score is not a guarantee of correctness."
        )

    st.subheader("♻️ Recommended Disposal")

    st.info(
        DISPOSAL_TIPS.get(
            label,
            "Follow your local waste-disposal guidelines."
        )
    )

    with st.expander("View scores for all six categories"):

        scores = np.asarray(result["scores"])

        for index in np.argsort(scores)[::-1]:

            category = class_names[int(index)]
            score = float(scores[int(index)])

            st.write(
                f"**{pretty_label(category)}:** "
                f"{score * 100:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, score))
            )

    if confidence < 0.60:
        st.warning(
            "This prediction may be unreliable. "
            "Try a clearer image with better lighting."
        )

    st.caption(
        "Previous validation accuracy was approximately 52%. "
        "Predictions may be incorrect."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "EcoSort | Waste Management Using CNN | Student Project"
)