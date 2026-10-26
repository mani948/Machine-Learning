
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

days = np.arange(500)
prices = 100 + 0.05 * days + 5 * np.sin(days / 10) + np.random.normal(0, 1, 500)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices.reshape(-1, 1))

X, y = [], []

for i in range(30, len(scaled)):
    X.append(scaled[i-30:i, 0])
    y.append(scaled[i, 0])

X = np.array(X).reshape(-1, 30, 1)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

predictions = model.predict(X_test, verbose=0)

predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

print("Predicted next price:", round(float(predictions[0][0]), 2))
print("Actual price:", round(float(actual[0][0]), 2))

plt.plot(actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("LSTM Price Prediction")
plt.show()
