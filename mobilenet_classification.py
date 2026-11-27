
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Dataset paths
train_dir = r"C:\interview\cats_and_dogs_filtered\train"
val_dir = r"C:\interview\cats_and_dogs_filtered\validation"

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary"
)

# Load validation images
val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary"
)

# Preprocess images for MobileNetV2
train_data = train_data.map(
    lambda x, y: (preprocess_input(x), y)
)

val_data = val_data.map(
    lambda x, y: (preprocess_input(x), y)
)

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build classification model
inputs = tf.keras.Input(shape=(160, 160, 3))

x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Evaluate model
loss, accuracy = model.evaluate(val_data)

print("Validation Accuracy:", accuracy * 100, "%")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Image Classification")
plt.legend()
plt.show()

# Save model
model.save("cat_dog_mobilenet.keras")

print("Model saved successfully!")
