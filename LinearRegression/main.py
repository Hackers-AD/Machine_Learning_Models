import numpy as np
from .models import LinearRegression
from model_selection import train_test_split
from datasets import make_regression
from metrics import mse_score, r2_score

if __name__ == "__main__":
    X, y = make_regression(n_samples=1000, n_features=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    model = LinearRegression(learning_rate=0.06)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    mse = mse_score(y_test, y_preds)
    score = r2_score(y_test, y_preds)

    print(f"MSE: {mse}, R2_SCORE: {score}")