"""Generate notebooks/transfer_value_analysis.ipynb with complete exploratory analysis and model evaluation."""

from pathlib import Path
import nbformat as nbf

def build_transfer_notebook():
    nb = nbf.v4.new_notebook()

    nb.metadata = {
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    }

    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# PremAI: Transfer Value Predictor Analysis
## Machine Learning Module 1 - Exploratory Data Analysis & Model Evaluation

This notebook documents the research, exploratory data analysis, and model validation for the **Transfer Value Predictor** module of the **Football Intelligence Platform (PremAI)**.

### Objectives:
1. **Dataset Overview**: Inspect the volume, attributes, and chronological temporal split (2017/18 through 2023/24).
2. **Target Distribution**: Analyze `market_value_eur` skewness and demonstrate the effectiveness of the `log1p` target transformation.
3. **Missing Value & Correlation Analysis**: Identify missingness patterns and relationships between demographic, financial, and on-pitch metrics.
4. **Model Comparison**: Compare Linear Regression, Random Forest, and XGBoost under raw vs. log-transformed target regimes.
5. **Error Analysis & Feature Importance**: Unpack model predictions on the unseen 2023/24 test set, identifying top predictive signals and residual patterns across player tiers.
"""))

    # Cell 1: Setup & Imports
    cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 5.5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['grid.linestyle'] = '--'

def format_eur(val):
    if abs(val) >= 1_000_000:
        return f"€{val / 1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"€{val / 1_000:.1f}K"
    return f"€{val:.0f}"

print("Environment setup and imports complete.")
"""))

    # Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Overview & Chronological Split

The dataset `data/processed/transfer/transfer_value_dataset.csv` combines Transfermarkt player valuations, squad valuations, and seasonal performance statistics across 7 seasons.

- **Train**: 2017/18 through 2021/22 (5 seasons)
- **Validation**: 2022/23 (1 season)
- **Test**: 2023/24 (1 season, held out for final evaluation)
"""))

    cells.append(nbf.v4.new_code_cell("""data_path = Path('../data/processed/transfer/transfer_value_dataset.csv')
if not data_path.exists():
    data_path = Path('data/processed/transfer/transfer_value_dataset.csv')

df = pd.read_csv(data_path)
print(f"Total dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns\\n")

print("Chronological split counts:")
for grp, cnt in df['split_group'].value_counts().items():
    print(f"  {grp.upper():<6}: {cnt:>6,} rows ({cnt / len(df) * 100:>5.1f}%)")

df[['player_name', 'club_name', 'season', 'age', 'primary_position', 'market_value_eur']].head(5)
"""))

    # Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Target Distribution: Raw vs. Log-Transformed

Player transfer valuations span several orders of magnitude (from €25,000 to €180,000,000). We compare the raw distribution against `log1p(market_value_eur)`.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw distribution
mv_millions = df['market_value_eur'] / 1e6
axes[0].hist(mv_millions, bins=60, color='#1f77b4', edgecolor='black', alpha=0.75)
axes[0].set_title('Raw Target: market_value_eur (Millions €)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Market Value (€ Millions)')
axes[0].set_ylabel('Player Seasons (Log scale)')
axes[0].set_yscale('log')

# Log1p distribution
log_mv = np.log1p(df['market_value_eur'])
axes[1].hist(log_mv, bins=50, color='#2ca02c', edgecolor='black', alpha=0.75)
axes[1].set_title('Transformed Target: log1p(market_value_eur)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('log1p(Market Value in EUR)')
axes[1].set_ylabel('Player Seasons')

plt.tight_layout()
plt.show()

print(f"Raw Target Skewness:        {df['market_value_eur'].skew():.2f}")
print(f"Transformed Target Skewness: {log_mv.skew():.2f}")
print(f"Mean Value:                 {format_eur(df['market_value_eur'].mean())}")
print(f"Median Value:               {format_eur(df['market_value_eur'].median())}")
print(f"Max Value:                  {format_eur(df['market_value_eur'].max())}")
"""))

    # Section 3
    cells.append(nbf.v4.new_markdown_cell("""## 3. Missing-Value Analysis

We analyze missing values across all predictive candidate features.
"""))

    cells.append(nbf.v4.new_code_cell("""candidate_features = [
    'age', 'height_cm', 'contract_years_left', 'club_market_value_eur',
    'minutes_played', 'appearances', 'goals', 'assists', 'yellow_cards',
    'red_cards', 'goals_per90', 'assists_per90', 'shots_per90', 'npxg_per90',
    'prev_market_value_eur', 'primary_position', 'sub_position', 'preferred_foot'
]

null_counts = df[candidate_features].isnull().sum()
null_pcts = (null_counts / len(df)) * 100
missing_summary = pd.DataFrame({'Missing Rows': null_counts, 'Missing %': null_pcts})
missing_summary = missing_summary[missing_summary['Missing Rows'] > 0].sort_values(by='Missing %', ascending=False)
missing_summary
"""))

    # Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Distributions & Categoricals

We observe the composition of player positions and age profiles.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pos_counts = df['primary_position'].value_counts()
axes[0].bar(pos_counts.index, pos_counts.values, color='#4575b4', edgecolor='black', alpha=0.8)
axes[0].set_title('Primary Position Distribution', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')

axes[1].hist(df['age'].dropna(), bins=22, color='#74add1', edgecolor='black', alpha=0.8)
axes[1].set_title('Age Distribution (Mean = {:.1f} yrs)'.format(df['age'].mean()), fontsize=12, fontweight='bold')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()
"""))

    # Section 5
    cells.append(nbf.v4.new_markdown_cell("""## 5. Correlation Analysis

We calculate Pearson correlations for numerical metrics against `market_value_eur`.
"""))

    cells.append(nbf.v4.new_code_cell("""corr_cols = [
    'market_value_eur', 'prev_market_value_eur', 'club_market_value_eur',
    'age', 'contract_years_left', 'minutes_played', 'appearances',
    'goals', 'assists', 'shots_per90', 'npxg_per90'
]

corr_matrix = df[corr_cols].corr()

print("Correlation with market_value_eur:")
for feat, val in corr_matrix['market_value_eur'].sort_values(ascending=False).items():
    if feat != 'market_value_eur':
        print(f"  {feat:<25}: {val:+.4f}")

fig, ax = plt.subplots(figsize=(9, 7))
cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-0.2, vmax=1.0)
fig.colorbar(cax, shrink=0.8)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha='left')
ax.set_yticklabels(corr_cols)
plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
"""))

    # Section 6
    cells.append(nbf.v4.new_markdown_cell("""## 6. Model Training & Validation Comparison

We trained three distinct regressors under both raw and `log1p` target transformations:
1. **Linear Regression** (OLS baseline)
2. **Random Forest Regressor** (100 estimators, max_depth=16)
3. **XGBoost Regressor** (150 estimators, learning_rate=0.08, max_depth=6)

All models were evaluated on the **2022/23 validation set** to make the final model selection.
"""))

    cells.append(nbf.v4.new_code_cell("""comp_path = Path('../models/transfer/model_comparison.csv')
if not comp_path.exists():
    comp_path = Path('models/transfer/model_comparison.csv')

comp_df = pd.read_csv(comp_path)
comp_df
"""))

    # Section 7
    cells.append(nbf.v4.new_markdown_cell("""### Model Selection Verdict:
- **Selected Model**: **XGBoost with log1p target transformation**
- **Validation MAE**: **€1.56M** (Lowest among all candidates)
- **Why log1p target?**: The logarithmic transformation squashes extreme positive skewness, allowing tree-based gradient boosting to model relative percentage differences across player tiers rather than over-focusing exclusively on the mega-value tail.
- **Why Linear Regression with log1p failed**: In OLS, linear combinations of one-hot encoded categories and unconstrained coefficients produced occasional large positive tail outputs (>25.0) which, when inverted via `expm1()`, yielded astronomically huge predictions and a negative R².
"""))

    # Section 8
    cells.append(nbf.v4.new_markdown_cell("""## 7. Feature Importance of the Selected Model

Let us inspect the relative feature importances from `models/transfer/feature_importance.csv`.
"""))

    cells.append(nbf.v4.new_code_cell("""fi_path = Path('../models/transfer/feature_importance.csv')
if not fi_path.exists():
    fi_path = Path('models/transfer/feature_importance.csv')

fi_df = pd.read_csv(fi_path)
top12 = fi_df.head(12)

plt.figure(figsize=(10, 5.5))
plt.barh(top12['feature'][::-1], top12['importance'][::-1], color='#3182bd', edgecolor='black')
plt.xlabel('Importance')
plt.title('Top 12 Most Important Features (XGBoost log1p)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

top12
"""))

    # Section 9
    cells.append(nbf.v4.new_markdown_cell("""## 8. Error Analysis on the Unseen 2023/24 Test Set

We load `models/transfer/test_predictions.csv` to analyze residuals, top overvalued players, and top undervalued players.
"""))

    cells.append(nbf.v4.new_code_cell("""preds_path = Path('../models/transfer/test_predictions.csv')
if not preds_path.exists():
    preds_path = Path('models/transfer/test_predictions.csv')

test_preds = pd.read_csv(preds_path)

def error_table(subset):
    tbl = subset[['player_name', 'club_name', 'primary_position', 'actual_market_value_eur', 'predicted_market_value_eur', 'absolute_error_eur', 'percentage_error']].copy()
    tbl['actual_market_value_eur'] = tbl['actual_market_value_eur'].apply(format_eur)
    tbl['predicted_market_value_eur'] = tbl['predicted_market_value_eur'].apply(format_eur)
    tbl['absolute_error_eur'] = tbl['absolute_error_eur'].apply(format_eur)
    tbl['percentage_error'] = tbl['percentage_error'].apply(lambda x: f"{x:.1f}%")
    return tbl

print("Top 5 Most Undervalued by Model (Actual > Predicted):")
display(error_table(test_preds.sort_values(by='signed_error_eur', ascending=False).head(5)))

print("\\nTop 5 Most Overvalued by Model (Predicted > Actual):")
display(error_table(test_preds.sort_values(by='signed_error_eur', ascending=True).head(5)))
"""))

    # Section 10
    cells.append(nbf.v4.new_markdown_cell("""## 9. Real-World Player Test Demonstrations

We invoke the production prediction interface `predict_transfer_value` from `backend.ml.transfer.predict` on key 2023/24 Premier League stars.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
sys.path.append('..')
from backend.ml.transfer.predict import predict_transfer_value

stars = ['Erling Haaland', 'Bukayo Saka', 'William Saliba', 'Alexander Isak', 'Declan Rice', 'Cole Palmer']
sub_df = df[(df['split_group'] == 'test') & (df['player_name'].isin(stars))].drop_duplicates(subset=['player_name'])

predictions = predict_transfer_value(sub_df)

demo_rows = []
for (_, row), pred in zip(sub_df.iterrows(), predictions):
    act = row['market_value_eur']
    pred_val = pred['predicted_market_value_eur']
    err = abs(act - pred_val)
    pct = (err / act) * 100 if act > 0 else 0
    demo_rows.append({
        'Player': row['player_name'],
        'Club': row['club_name'],
        'Position': row['primary_position'],
        'Actual Value': format_eur(act),
        'Predicted Value': format_eur(pred_val),
        'Abs Error': format_eur(err),
        'Pct Error': f"{pct:.1f}%"
    })

pd.DataFrame(demo_rows)
"""))

    # Section 11
    cells.append(nbf.v4.new_markdown_cell("""## 10. Conclusions & Application Integration

### Key Insights:
1. **Strong Generalization**: The final selected model achieves **R² = 0.8986** on completely unseen 2023/24 test data with a **Mean Absolute Error of €1.71M** and **RMSE of €4.28M**.
2. **Foundational Anchors**: Previous market valuation (`prev_market_value_eur`, 57.8%) and club valuation tier (`club_market_value_eur`, 15.9%) establish the baseline valuation tier, while player age (3.5%), minutes played (3.4%), goals (3.3%), and appearances (2.8%) modulate value adjustments up or down.
3. **Application Integration**: The model and preprocessor pipeline are packaged into `models/transfer/transfer_value_model.joblib`. The prediction service `predict_transfer_value(player_features)` exposes a clean API ready to be called by FastAPI in Step 6.
"""))

    nb.cells = cells

    out_path = Path("notebooks/transfer_value_analysis.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated notebook at: {out_path.resolve()}")

if __name__ == "__main__":
    build_transfer_notebook()
