
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Small sample dataset
reviews = [
    "this movie is amazing",
    "this film is wonderful",
    "i really love this movie",
    "the acting is excellent",
    "what a fantastic film",
    "this movie is great",
    "this movie is terrible",
    "i hate this film",
    "the acting is bad",
    "this movie is boring",
    "what a horrible film",
    "the movie is disappointing"
]

# 1 = Positive, 0 = Negative
labels = np.array([
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
])

# Convert text into numbers
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(sequences, maxlen=6, padding="post")

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16),
    LSTM(16),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(X, labels, epochs=100, verbose=0)

# Test with new reviews
test_reviews = [
    "this movie is wonderful",
    "this film is horrible"
]

test_sequences = tokenizer.texts_to_sequences(test_reviews)
test_data = pad_sequences(
    test_sequences, maxlen=6, padding="post"
)

predictions = model.predict(test_data, verbose=0)

for review, prediction in zip(test_reviews, predictions):
    sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"
    print("Review:", review)
    print("Predicted sentiment:", sentiment)
    print()
