from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load dataset (using newsgroups as example for text classification)
data = fetch_20newsgroups(subset='all', categories=['rec.sport.hockey','sci.space'])
X, y = data.data, data.target

# 2. Preprocess text
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# 4. Train models
nb = MultinomialNB().fit(X_train, y_train)
lr = LogisticRegression(max_iter=200).fit(X_train, y_train)

# 5. Predictions
y_pred_nb = nb.predict(X_test)
y_pred_lr = lr.predict(X_test)

# 6. Evaluation
for name, preds in {"Naive Bayes": y_pred_nb, "Logistic Regression": y_pred_lr}.items():
    print(name)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='macro'))
    print("Recall:", recall_score(y_test, preds, average='macro'))
    print("F1-score:", f1_score(y_test, preds, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("-"*40)