from dotenv import load_dotenv
import torch
import streamlit as st

from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection

# Hugging Face Token (keep it safe in real apps)
load_dotenv()
#HF_TOKEN = "your_huggingface_token_here"

# Load model and processor from HuggingFace
@st.cache_resource
def load_model():
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", 
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", 
    )
    return processor, model

# Draw bounding boxes
def draw_boxes(image, outputs, threshold=0.9):
    draw = ImageDraw.Draw(image)
    logits = outputs.logits.softmax(-1)[0]
    boxes = outputs.pred_boxes[0]

    for logit, box in zip(logits, boxes):
        score, label = logit.max(0)
        if score > threshold and label != 91:  # 91 is class 'no-object'
            box = box.detach().cpu().numpy()
            w, h = image.size
            x_center, y_center, width, height = box
            x0 = (x_center - width / 2) * w
            y0 = (y_center - height / 2) * h
            x1 = (x_center + width / 2) * w
            y1 = (y_center + height / 2) * h
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((x0, y0), f"{model.config.id2label[label.item()]}: {score:.2f}", fill="red")
    return image

# Streamlit UI
st.title("🔍 Object Detection App (Hugging Face + Streamlit)")
st.write("Upload an image and detect objects using DETR (facebook/detr-resnet-50)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processor, model = load_model()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    processed_image = image.copy()
    processed_image = draw_boxes(processed_image, outputs)

    st.image(processed_image, caption="Detected Objects", use_column_width=True)
