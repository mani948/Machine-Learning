
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
tf.random.set_seed(42)

X = []
y = []

activities = ["walking", "running", "sitting"]

for activity in activities:
    for _ in range(300):
        if activity == "walking":
            signal = np.sin(np.linspace(0, 6*np.pi, 30))
        elif activity == "running":
            signal = 2 * np.sin(np.linspace(0, 10*np.pi, 30))
        else:
            signal = np.zeros(30)

        signal = signal + np.random.normal(0, 0.15, 30)
        X.append(signal)
        y.append(activity)

X = np.array(X).reshape(-1, 30, 1)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(accuracy * 100, 2), "%")

sample = X_test[:1]
prediction = model.predict(sample, verbose=0)
predicted_class = encoder.inverse_transform(
    [np.argmax(prediction)]
)[0]

actual_class = encoder.inverse_transform([y_test[0]])[0]

print("Predicted activity:", predicted_class)
print("Actual activity:", actual_class)
