import numpy as np
from model_selection import train_test_split
from .models import KNNClassifier
from metrics import accuracy_score
from datasets import make_classification

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    clf = KNNClassifier(k=10)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_preds)
    print(f"Accuracy : {accuracy}")