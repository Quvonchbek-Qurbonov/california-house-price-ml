import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from app.core.config import settings


df = pd.read_csv(settings.DATASET_DIR / settings.DATASET)

X = df.drop(columns=["Class"])


numeric_features = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


def log_amount(X):
    X = X.copy()

    if isinstance(X, np.ndarray):
        X = X.astype(float)
        X[:, -1] = np.log1p(X[:, -1])
        return X

    X["Amount"] = np.log1p(X["Amount"])
    return X


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log_amount", FunctionTransformer(
        log_amount,
        feature_names_out="one-to-one"
    )),
    ("scaler", StandardScaler()),
])


preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features)
    ]
)


X_processed = preprocessor.fit_transform(X)

feature_names = preprocessor.get_feature_names_out()

X_processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names,
    index=X.index
)