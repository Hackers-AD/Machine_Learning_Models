from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from models import LogisticRegression


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_preds = model.predict(X_test)

accuracy = model.accuracy_score(y_test, y_preds)
print(f"Model Accuracy: {accuracy}")