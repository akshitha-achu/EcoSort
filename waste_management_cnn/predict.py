"""
Predict whether a single image is wet_waste or plastic.
Run after training:
    python predict.py path/to/image.jpg
"""
import os
import sys
import numpy as np
import tensorflow as tf

MODEL_PATH = "waste_classifier.keras"
LABELS_PATH = "class_names.txt"
IMG_SIZE = (180, 180)

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        raise SystemExit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(MODEL_PATH):
        print("Model not found. First run: python train_model.py")
        raise SystemExit(1)
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        raise SystemExit(1)

    if os.path.isfile(LABELS_PATH):
        with open(LABELS_PATH, encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = ["plastic", "wet_waste"]

    model = tf.keras.models.load_model(MODEL_PATH)
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    score = float(model.predict(arr, verbose=0)[0][0])
    # With binary labels, sigmoid scores correspond to class index 1.
    idx = 1 if score >= 0.5 else 0
    confidence = score if idx == 1 else 1.0 - score
    print("\n--- Waste Classification Result ---")
    print(f"Image: {image_path}")
    print(f"Prediction: {class_names[idx].replace('_', ' ').title()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("-----------------------------------")

if __name__ == "__main__":
    main()
