"""Preprocessing pipelines for the Transfer Value Predictor module."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Model features
NUMERICAL_FEATURES = [
    "age",
    "height_cm",
    "contract_years_left",
    "club_market_value_eur",
    "minutes_played",
    "appearances",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "goals_per90",
    "assists_per90",
    "shots_per90",
    "npxg_per90",
    "prev_market_value_eur",
]

CATEGORICAL_FEATURES = [
    "primary_position",
    "sub_position",
    "preferred_foot",
]

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "market_value_eur"


def get_transfer_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer for Transfer Value Predictor.

    - Numerical: Median imputation + Standard scaling
    - Categorical: Most frequent imputation + OneHotEncoder(handle_unknown='ignore')
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def get_transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Extract readable feature names after ColumnTransformer transformation."""
    feature_names = []
    # 1. Numerical names
    feature_names.extend(NUMERICAL_FEATURES)

    # 2. Categorical one-hot names
    cat_transformer = fitted_preprocessor.named_transformers_["cat"]
    onehot_step = cat_transformer.named_steps["onehot"]
    cat_names = onehot_step.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(cat_names))

    return feature_names
