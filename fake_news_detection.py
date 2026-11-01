
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample news dataset
data = {
    "news": [
        "government announces new education policy",
        "scientists discover new technology",
        "new hospital opened in the city",
        "researchers develop cancer treatment",
        "school students achieve excellent results",
        "government launches new public scheme",
        "aliens secretly control the government",
        "miracle drink makes people immortal",
        "celebrity becomes invisible overnight",
        "secret machine predicts lottery numbers",
        "moon made entirely of cheese",
        "one tablet gives unlimited intelligence"
    ],

    # 1 = Real, 0 = Fake
    "label": [
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

# Separate input and output
X = df["news"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Build and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

# Predict new news
new_news = [
    "scientists discover new technology",
    "miracle tablet makes people immortal"
]

new_data = vectorizer.transform(new_news)
results = model.predict(new_data)

for news, result in zip(new_news, results):
    if result == 1:
        print("\nNews:", news)
        print("Prediction: Real News")
    else:
        print("\nNews:", news)
        print("Prediction: Fake News")
