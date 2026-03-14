from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# sample dataset
emails = [
    "Win money now",        # spam
    "Limited time offer",   # spam
    "Meeting tomorrow",     # not spam
    "Project deadline",     # not spam
]
labels = [1, 1, 0, 0]  # 1 = spam, 0 = not spam

# convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# make a prediction
test_email = ["Win a free prize"]
X_test = vectorizer.transform(test_email)
prediction = model.predict(X_test)

print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")