import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin1')

# Keep required columns
df = df[['v1','v2']]
df.columns = ['text','target']

# Convert labels
df['text'] = df['text'].map({'ham':0,'spam':1})

# Split data
from sklearn.model_selection import train_test_split

X = df['target']
y = df['text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_spam(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "Spam" if result == 1 else "Not Spam"

# User input loop
while True:
    msg = input("Enter message: ")
    print(predict_spam(msg))