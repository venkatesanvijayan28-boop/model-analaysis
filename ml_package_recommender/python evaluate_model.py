import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ==========================
# SETTINGS
# ==========================

IMG_SIZE = 224
BATCH_SIZE = 16

# ==========================
# LOAD MODEL
# ==========================

model = tf.keras.models.load_model("models/brain_tumor_model.h5")

# ==========================
# LOAD TEST DATA
# ==========================

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "dataset/Testing",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==========================
# EVALUATE MODEL
# ==========================

loss, accuracy = model.evaluate(test_data)

print("\n==============================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy*100:.2f}%")
print("==============================\n")

# ==========================
# PREDICTIONS
# ==========================

predictions = model.predict(test_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data.classes

class_labels = list(test_data.class_indices.keys())

# ==========================
# CLASSIFICATION REPORT
# ==========================

print("Classification Report:\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_labels
    )
)

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix:\n")

print(cm)