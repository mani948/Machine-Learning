
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", round(accuracy * 100, 2), "%")

prediction = model.predict(X_test[:1], verbose=0)[0][0]
print("Prediction score:", round(float(prediction), 4))
print("Predicted class:", data.target_names[int(prediction >= 0.5)])
print("Actual class:", data.target_names[y_test[0]])
