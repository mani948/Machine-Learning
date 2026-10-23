
import os
import tensorflow as tf
import matplotlib.pyplot as plt

base = r"C:\interview\ML\cats_and_dogs_filtered"

train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base, "train"),
    image_size=(150, 150),
    batch_size=32
)

val = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base, "validation"),
    image_size=(150, 150),
    batch_size=32
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation="relu",
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=5)

images, labels = next(iter(val))
prediction = model.predict(images[:1])[0][0]

print("Actual:", "Dog" if labels[0].numpy() == 1 else "Cat")
print("Predicted:", "Dog" if prediction >= 0.5 else "Cat")

plt.imshow(images[0].numpy().astype("uint8"))
plt.show()

