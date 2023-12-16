from ..model_selection import train_test_split
from models import LogisticRegression
from ..datasets import make_classification


X, y = make_classification(n_samples=1000, n_features=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_preds = model.predict(X_test)

accuracy = model.accuracy_score(y_test, y_preds)
print(f"Model Accuracy: {accuracy}")