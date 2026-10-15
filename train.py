import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.random.uniform([0, 40, 0], [10, 100, 100], size=(500, 3))
y = ((X[:, 0] >= 3) & (X[:, 1] >= 60) & (X[:, 2] >= 40)).astype(int)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X, y)
joblib.dump(model, "model.pkl")
print("Model trained and saved successfully.")
