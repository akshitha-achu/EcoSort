import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# ECOSORT - IMPROVED WASTE CLASSIFICATION
# MobileNetV2 Transfer Learning
# ==========================================

DATA_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

MODEL_PATH = "waste_classifier.keras"
LABELS_PATH = "class_names.txt"

EXPECTED_CLASSES = [
    "e-waste",
    "glass",
    "metal",
    "paper",
    "plastic",
    "wet_waste",
]

SEED = 42

# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR}"
    )

for category in EXPECTED_CLASSES:
    path = os.path.join(DATA_DIR, category)

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing category folder: {path}"
        )

print("\nLoading dataset...")

# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=EXPECTED_CLASSES,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=EXPECTED_CLASSES,
)

class_names = train_ds.class_names

print("\nClass order:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ==========================================
# CHECK CLASS DISTRIBUTION
# ==========================================

class_counts = np.zeros(len(class_names), dtype=int)

for _, labels in train_ds:
    label_array = labels.numpy().astype(int)
    class_counts += np.bincount(
        label_array,
        minlength=len(class_names)
    )

print("\nTraining images per class:")

for name, count in zip(class_names, class_counts):
    print(f"{name}: {count}")

if np.any(class_counts == 0):
    raise ValueError(
        "At least one class has no training images."
    )

# Class weights help when categories have
# different numbers of training images.
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.repeat(
        np.arange(len(class_names)),
        class_counts
    ),
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

# ==========================================
# OPTIMIZE DATA PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(
    500,
    seed=SEED
).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# DATA AUGMENTATION
# ==========================================

augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ==========================================
# LOAD PRETRAINED MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Initially freeze pretrained layers.
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = augmentation(inputs)

# MobileNetV2 expects pixels scaled to [-1, 1].
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nModel architecture:")
model.summary()

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ==========================================
# TRAIN CLASSIFICATION HEAD
# ==========================================

print("\nStarting transfer learning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# ==========================================
# FINE-TUNE TOP LAYERS
# ==========================================

print("\nStarting fine-tuning...")

base_model.trainable = True

# Freeze most pretrained layers.
# Fine-tune only the last 30 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=fine_tune_callbacks,
    class_weight=class_weights,
)

# ==========================================
# SAVE FINAL MODEL AND LABELS
# ==========================================

model.save(MODEL_PATH)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

print("\n====================================")
print("TRAINING COMPLETED")
print("====================================")

print(f"Model saved: {MODEL_PATH}")
print(f"Labels saved: {LABELS_PATH}")

print(
    "Final training accuracy: "
    f"{history_fine.history['accuracy'][-1] * 100:.2f}%"
)

print(
    "Final validation accuracy: "
    f"{history_fine.history['val_accuracy'][-1] * 100:.2f}%"
)

print("\nSix-class EcoSort model is ready!")