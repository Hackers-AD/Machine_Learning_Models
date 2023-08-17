import numpy as np
from model_selection import train_test_split
from models import KNNClassifier
from metrics import predict_accuracy

if __name__ == "__main__":
    X = np.random.uniform(0, 1, (1000, 8))
    y = np.random.randint(0, 5, len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    clf = KNNClassifier(k=10)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    accuracy = predict_accuracy(y_test, y_preds)
    print(f"Accuracy : {accuracy}")