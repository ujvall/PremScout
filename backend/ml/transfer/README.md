# Transfer Value Predictor (ML Module 1)

This module predicts a football player's estimated transfer market valuation based on demographic traits, contract status, club tier, and seasonal performance statistics.

---

## 📁 Module Architecture

```text
backend/ml/transfer/
├── __init__.py          # Package interface
├── preprocessing.py     # ColumnTransformer (imputation + scaling + one-hot encoding)
├── evaluate.py          # MAE, RMSE, R2, MAPE, and currency formatters
├── train.py             # Model training, comparison, selection, and visualization
├── predict.py           # Clean inference function (predict_transfer_value)
└── README.md            # Module documentation
```

---

## 🚀 How to Run Training

From the project root:

```powershell
.\venv\Scripts\python.exe -m backend.ml.transfer.train
```

This will:
1. Load `data/processed/transfer/transfer_value_dataset.csv`.
2. Split into Train (2017/18–2021/22), Validation (2022/23), and Test (2023/24).
3. Train 6 model variants (Linear Regression, Random Forest, XGBoost under both raw and log1p targets).
4. Select the best model strictly using the Validation set MAE.
5. Evaluate on the untouched 2023/24 test set.
6. Save:
   - `models/transfer/transfer_value_model.joblib`
   - `models/transfer/model_metadata.json`
   - `models/transfer/model_comparison.csv`
   - `models/transfer/test_predictions.csv`
   - `models/transfer/feature_importance.csv`
   - `models/transfer/plots/*.png`

---

## 🔮 How to Run Inference

```python
from backend.ml.transfer.predict import predict_transfer_value

player = {
    "age": 24.5,
    "height_cm": 185.0,
    "contract_years_left": 3.0,
    "club_market_value_eur": 850_000_000,
    "minutes_played": 2800,
    "appearances": 34,
    "goals": 18,
    "assists": 8,
    "yellow_cards": 3,
    "red_cards": 0,
    "goals_per90": 0.58,
    "assists_per90": 0.26,
    "shots_per90": 2.8,
    "npxg_per90": 0.50,
    "prev_market_value_eur": 45_000_000,
    "primary_position": "Attack",
    "sub_position": "Centre-Forward",
    "preferred_foot": "right",
}

prediction = predict_transfer_value(player)
print(prediction)
# Output: {'predicted_market_value_eur': 65000000.0, 'formatted_market_value': '€65.00M', ...}
```
