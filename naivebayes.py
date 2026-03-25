from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# sample dataset
emails = [
    "Win money now", 
    "Limited offer just for you", 
    "Meeting at 10am tomorrow", 
    "Project deadline extended", 
    "Congratulations you won a prize"
]
labels = [1, 1, 0, 0, 1]  # 1 = Spam, 0 = Not Spam

# convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# test prediction
test_email = ["Free money offer"]
X_test = vectorizer.transform(test_email)
prediction = model.predict(X_test)

print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")