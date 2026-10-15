# Original OLS models with validation, a mean/Ridge comparison and error plots.
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import matplotlib
matplotlib.use("Agg")
try:
    from IPython.display import display
except ImportError:
    display = print
import pandas as pd

matches = pd.read_csv("matches_detailed.csv")
features = pd.read_csv("match_prediction_features_X.csv")
targets = pd.read_csv("match_prediction_targets_y.csv")

print("Total Matches:", matches.shape)
print("Features Dataset:", features.shape)
print("Targets Dataset:", targets.shape)

display(matches.head())


print("FEATURE COLUMNS:")
print(features.columns.tolist())

print("\nTARGET COLUMNS:")
print(targets.columns.tolist())


for i, column in enumerate(features.columns, start=1):
    print(f"{i}. {column}")


print("Total match records:", len(matches))
print("Total feature records:", len(features))
print("Total target records:", len(targets))

print("\nUnique match IDs:")
print("Matches:", matches["match_id"].nunique())
print("Features:", features["match_id"].nunique())
print("Targets:", targets["match_id"].nunique())

print("\nMissing values:")
print(features.isnull().sum().to_string())


print("ALL AVAILABLE FEATURE COLUMNS:\n")

for i, column in enumerate(features.columns, 1):
    print(f"{i}. {column}")


# Create a copy of the match results
results = targets.copy()

# Calculate goal difference
results["goal_difference"] = (
    results["home_score"] - results["away_score"]
)

# Display first 10 matches
display(
    results[
        ["match_id", "home_score",
         "away_score", "goal_difference"]
    ].head(10)
)


# Display all available explanatory variables
for i, col in enumerate(features.columns, 1):
    print(i, col)


# Show feature columns from number 25 onwards

for i, col in enumerate(features.columns[24:], start=25):
    print(f"{i}. {col}")


import pandas as pd

# Merge pre-match features with actual match results
df = features.merge(
    targets[["match_id", "home_score", "away_score"]],
    on="match_id",
    how="inner",
    validate="one_to_one"
)

# Create regression dataset
lr21 = pd.DataFrame()

lr21["match_id"] = df["match_id"]
lr21["home_team"] = df["home_team_name"]
lr21["away_team"] = df["away_team_name"]

# Exactly 8 explanatory variables
lr21["ranking_diff"] = (
    df["away_fifa_rank"] - df["home_fifa_rank"]
)

lr21["elo_diff"] = df["home_elo"] - df["away_elo"]

lr21["squad_value_diff"] = (
    df["home_squad_total_value_eur"]
    - df["away_squad_total_value_eur"]
) / 1_000_000

lr21["squad_age_diff"] = (
    df["home_squad_avg_age"] - df["away_squad_avg_age"]
)

lr21["rest_days_diff"] = (
    df["home_rest_days"] - df["away_rest_days"]
)

lr21["previous_goals_scored_diff"] = (
    df["home_prev_avg_goals_scored"]
    - df["away_prev_avg_goals_scored"]
)

lr21["previous_goals_conceded_diff"] = (
    df["home_prev_avg_goals_conceded"]
    - df["away_prev_avg_goals_conceded"]
)

lr21["host_advantage"] = (
    df["home_is_host"].astype(int)
    - df["away_is_host"].astype(int)
)

# Target variable
lr21["goal_difference"] = (
    df["home_score"] - df["away_score"]
)

# Verify dataset
assert len(lr21) == 104
assert lr21["match_id"].nunique() == 104

print("Dataset shape:", lr21.shape)
display(lr21.head())


from pathlib import Path
Path("analysis_outputs").mkdir(exist_ok=True)
lr21.to_csv("analysis_outputs/Linear_Regression_2_1.csv", index=False)
print("Saved analysis_outputs/Linear_Regression_2_1.csv")


print("Missing values:")
print(lr21.isnull().sum())

print("\nTotal rows:", len(lr21))
print("Unique matches:", lr21["match_id"].nunique())


import pandas as pd

# Combine features and actual match scores
df = features.merge(
    targets[["match_id", "home_score", "away_score"]],
    on="match_id",
    how="inner",
    validate="one_to_one"
)

# Home team observations
home = pd.DataFrame({
    "match_id": df["match_id"],
    "team": df["home_team_name"],
    "opponent": df["away_team_name"],
    "team_fifa_rank": df["home_fifa_rank"],
    "team_elo": df["home_elo"],
    "team_prev_avg_goals_scored":
        df["home_prev_avg_goals_scored"],
    "opponent_prev_avg_goals_conceded":
        df["away_prev_avg_goals_conceded"],
    "team_prev_avg_shots_on_target":
        df["home_prev_avg_shots_on_target"],
    "team_prev_avg_possession":
        df["home_prev_avg_possession"],
    "opponent_prev_avg_saves":
        df["away_prev_avg_saves"],
    "is_knockout": df["is_knockout"],
    "goals_scored": df["home_score"]
})

# Away team observations
away = pd.DataFrame({
    "match_id": df["match_id"],
    "team": df["away_team_name"],
    "opponent": df["home_team_name"],
    "team_fifa_rank": df["away_fifa_rank"],
    "team_elo": df["away_elo"],
    "team_prev_avg_goals_scored":
        df["away_prev_avg_goals_scored"],
    "opponent_prev_avg_goals_conceded":
        df["home_prev_avg_goals_conceded"],
    "team_prev_avg_shots_on_target":
        df["away_prev_avg_shots_on_target"],
    "team_prev_avg_possession":
        df["away_prev_avg_possession"],
    "opponent_prev_avg_saves":
        df["home_prev_avg_saves"],
    "is_knockout": df["is_knockout"],
    "goals_scored": df["away_score"]
})

# Combine home and away teams
lr22 = pd.concat([home, away], ignore_index=True)

# Verify dataset
assert len(lr22) == 208
assert lr22["match_id"].nunique() == 104

print("Dataset shape:", lr22.shape)
display(lr22.head(10))


print("Total rows:", len(lr22))
print("Unique matches:", lr22["match_id"].nunique())

print("\nMissing values:")
print(lr22.isnull().sum())


from pathlib import Path
Path("analysis_outputs").mkdir(exist_ok=True)
lr22.to_csv("analysis_outputs/Linear_Regression_2_2.csv", index=False)
print("Saved analysis_outputs/Linear_Regression_2_2.csv")


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
import pandas as pd

# Select exactly 8 explanatory variables
x_columns = [
    "ranking_diff",
    "elo_diff",
    "squad_value_diff",
    "squad_age_diff",
    "rest_days_diff",
    "previous_goals_scored_diff",
    "previous_goals_conceded_diff",
    "host_advantage"
]

X = lr21[x_columns]
y = lr21["goal_difference"]

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model21 = LinearRegression()
model21.fit(X_train, y_train)

# Predict
y_pred = model21.predict(X_test)

# Evaluate
print("LINEAR REGRESSION 2.1 RESULTS")
print("R²:", round(r2_score(y_test, y_pred), 4))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(
    mean_squared_error(y_test, y_pred)
), 4))


from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import numpy as np

# Exactly 8 explanatory variables
x_columns_22 = [
    "team_fifa_rank",
    "team_elo",
    "team_prev_avg_goals_scored",
    "opponent_prev_avg_goals_conceded",
    "team_prev_avg_shots_on_target",
    "team_prev_avg_possession",
    "opponent_prev_avg_saves",
    "is_knockout"
]

X22 = lr22[x_columns_22]
y22 = lr22["goals_scored"]

# Keep both teams from each match together
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X22, y22, groups=lr22["match_id"])
)

X_train22 = X22.iloc[train_idx]
X_test22 = X22.iloc[test_idx]
y_train22 = y22.iloc[train_idx]
y_test22 = y22.iloc[test_idx]

# Train model
model22 = LinearRegression()
model22.fit(X_train22, y_train22)

# Make predictions
y_pred22 = model22.predict(X_test22)

# Evaluate model
print("LINEAR REGRESSION 2.2 RESULTS")
print("R²:", round(r2_score(y_test22, y_pred22), 4))
print("MAE:", round(mean_absolute_error(y_test22, y_pred22), 4))
print("RMSE:", round(np.sqrt(
    mean_squared_error(y_test22, y_pred22)
), 4))

print("\nTraining rows:", len(X_train22))
print("Testing rows:", len(X_test22))


import matplotlib.pyplot as plt

# Linear Regression 2.1 graph
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

limits = [
    min(y_test.min(), y_pred.min()),
    max(y_test.max(), y_pred.max())
]

plt.plot(limits, limits, "r--",
         label="Perfect prediction")

plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")
plt.title("Linear Regression 2.1: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("analysis_outputs/Regression_2_1_Graph.png", dpi=300)
plt.show()


plt.figure(figsize=(7, 5))
plt.scatter(y_test22, y_pred22, alpha=0.7)

limits = [
    min(y_test22.min(), y_pred22.min()),
    max(y_test22.max(), y_pred22.max())
]

plt.plot(limits, limits, "r--",
         label="Perfect prediction")

plt.xlabel("Actual Goals Scored")
plt.ylabel("Predicted Goals Scored")
plt.title("Linear Regression 2.2: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("analysis_outputs/Regression_2_2_Graph.png", dpi=300)
plt.show()


import pandas as pd

coefficients22 = pd.DataFrame({
    "Variable": x_columns_22,
    "Coefficient": model22.coef_
})

print("Linear Regression 2.2 Coefficients:")
display(coefficients22.round(4))

print("Intercept:", round(model22.intercept_, 4))


# Linear Regression 2.1 coefficients

coefficients21 = pd.DataFrame({
    "Variable": x_columns,
    "Coefficient": model21.coef_
})

print("Linear Regression 2.1 Coefficients:")
display(coefficients21.round(4))

print("Intercept:", round(model21.intercept_, 4))


# Check tournament dates and match status

print("First match date:", matches["date"].min())
print("Last match date:", matches["date"].max())

print("\nMatch status:")
print(matches["status"].value_counts())

print("\nCheck first five matches:")
display(matches[
    ["date", "home_team_name", "away_team_name",
     "home_score", "away_score"]
].head())


# Basic five-fold cross-validation using ONLY the original training rows.
from sklearn.model_selection import KFold, GroupKFold, cross_validate

scoring = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error", "r2": "r2"}

folds21 = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
training_match_ids22 = lr22.iloc[train_idx]["match_id"].reset_index(drop=True)
folds22 = list(GroupKFold(n_splits=5).split(
    X_train22, y_train22, groups=training_match_ids22))
for train_fold, validation_fold in folds22:
    assert set(training_match_ids22.iloc[train_fold]).isdisjoint(
        training_match_ids22.iloc[validation_fold])

cv21 = cross_validate(LinearRegression(), X_train, y_train,
                     cv=folds21, scoring=scoring)
cv22 = cross_validate(LinearRegression(), X_train22, y_train22,
                     cv=folds22, scoring=scoring)
cv_rows = []
for task, scores, folds in [("2.1", cv21, folds21), ("2.2", cv22, folds22)]:
    for fold, (training_rows, validation_rows) in enumerate(folds, 1):
        cv_rows.append({"task": task, "fold": fold,
                        "training_rows": len(training_rows),
                        "validation_rows": len(validation_rows),
                        "mae": -scores["test_mae"][fold-1],
                        "rmse": -scores["test_rmse"][fold-1],
                        "r2": scores["test_r2"][fold-1]})
cv_results = pd.DataFrame(cv_rows)
cv_summary = cv_results.groupby("task")[["mae", "rmse", "r2"]].mean()
cv_results.to_csv("analysis_outputs/cv_results.csv", index=False)
print("Five-fold training-only validation: arithmetic mean scores")
display(cv_summary.round(4))
print("Task 2.2 keeps both team rows from a match together in every fold.")



# Reproduce the report's descriptive statistics and figures.
# These describe all supplied records; they do not select predictors or models.
target_stats = pd.DataFrame({
    "Goal difference": lr21["goal_difference"].describe(),
    "Team goals": lr22["goals_scored"].describe()})
display(target_stats.round(4))
correlations21 = lr21[x_columns + ["goal_difference"]].corr()[
    "goal_difference"].drop("goal_difference").sort_values(ascending=False)
display(correlations21.round(4))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(lr21["goal_difference"], bins=np.arange(-4.5, 7.5, 1),
             color="0.4", edgecolor="white")
axes[0].set(xlabel="Match goal difference", ylabel="Frequency")
axes[1].hist(lr22["goals_scored"], bins=np.arange(-0.5, 8.5, 1),
             color="0.4", edgecolor="white")
axes[1].set(xlabel="Team goals scored", ylabel="Frequency")
fig.tight_layout()
fig.savefig("analysis_outputs/Report_Target_Distributions.png", dpi=220)
plt.show()

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.barh(correlations21.index, correlations21.values, color="0.4")
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.7)
ax.set_xlabel("Pearson correlation with goal difference (all 104 matches)")
fig.tight_layout()
fig.savefig("analysis_outputs/Report_Goal_Difference_Correlations.png", dpi=220)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, actual, predicted, title in [
    (axes[0], y_test, y_pred, "2.1 Goal difference"),
    (axes[1], y_test22, y_pred22, "2.2 Team goals")]:
    ax.scatter(actual, predicted, color="0.25", alpha=0.7)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set(xlabel="Actual goals / goal difference", ylabel="Predicted", title=title)
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("analysis_outputs/Report_Test_Predictions.png", dpi=220)
plt.show()



# Compare models on the SAME original training folds and eight inputs.
# The mean is a benchmark; both required original OLS models remain intact.
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import json

ridge_alphas = [1.0, 10.0, 100.0]
candidate_specs = [("Mean", None, DummyRegressor(strategy="mean")),
                   ("OLS", None, LinearRegression())]
candidate_specs += [("Ridge", alpha,
                     make_pipeline(StandardScaler(), Ridge(alpha=alpha)))
                    for alpha in ridge_alphas]
comparison_training = [("2.1", X_train, y_train, folds21),
                       ("2.2", X_train22, y_train22, folds22)]
comparison_fold_rows, comparison_cv_rows = [], []
for task, train_x, train_y, folds in comparison_training:
    for name, alpha, estimator in candidate_specs:
        scores = cross_validate(estimator, train_x, train_y, cv=folds,
                                scoring=scoring, error_score="raise")
        for i, (fit_rows, valid_rows) in enumerate(folds):
            comparison_fold_rows.append({
                "task": task, "model": name, "alpha": alpha, "fold": i + 1,
                "training_rows": len(fit_rows), "validation_rows": len(valid_rows),
                "mae": -scores["test_mae"][i],
                "rmse": -scores["test_rmse"][i], "r2": scores["test_r2"][i]})
        comparison_cv_rows.append({
            "task": task, "model": name, "alpha": alpha,
            "cv_mae": -scores["test_mae"].mean(),
            "cv_rmse": -scores["test_rmse"].mean(),
            "cv_r2": scores["test_r2"].mean(),
            "cv_rmse_sd": scores["test_rmse"].std(ddof=1)})
comparison_folds = pd.DataFrame(comparison_fold_rows)
comparison_cv = pd.DataFrame(comparison_cv_rows)
selected_ridge_alphas = {}
for task in ["2.1", "2.2"]:
    choices = comparison_cv[(comparison_cv["task"] == task) &
                            (comparison_cv["model"] == "Ridge")]
    best = choices.sort_values(["cv_rmse", "alpha"]).iloc[0]
    selected_ridge_alphas[task] = float(best["alpha"])
    original = comparison_cv[(comparison_cv["task"] == task) &
                             (comparison_cv["model"] == "OLS")].iloc[0]
    assert np.allclose(original[["cv_mae", "cv_rmse", "cv_r2"]].astype(float),
                       cv_summary.loc[task, ["mae", "rmse", "r2"]])
comparison_folds.to_csv("analysis_outputs/comparison_cv_folds.csv", index=False)
comparison_cv.to_csv("analysis_outputs/comparison_cv_summary.csv", index=False)
# Freeze the training-only choice BEFORE evaluating the additional test predictions.
selection = {"ridge_alphas_tried": ridge_alphas,
             "selection_rule": "Lowest mean five-fold training CV RMSE",
             "selected_ridge_alphas": selected_ridge_alphas,
             "scaling": "StandardScaler fitted inside each training fold",
             "cv_interpretation": "Selection scores, not independent post-selection estimates",
             "test_history": "Original test outcomes were inspected in earlier drafts"}
Path("analysis_outputs/comparison_selection.json").write_text(
    json.dumps(selection, indent=2))

comparison_rows, comparison_predictions = [], {}
for task, train_x, train_y, test_x, test_y, original_model in [
    ("2.1", X_train, y_train, X_test, y_test, model21),
    ("2.2", X_train22, y_train22, X_test22, y_test22, model22)]:
    assert train_x.shape[1] == test_x.shape[1] == 8
    alpha = selected_ridge_alphas[task]
    estimators = [("Mean", None, DummyRegressor(strategy="mean")),
                  ("OLS", None, original_model),
                  ("Ridge", alpha, make_pipeline(StandardScaler(), Ridge(alpha=alpha)))]
    for name, penalty, estimator in estimators:
        if name != "OLS": estimator.fit(train_x, train_y)
        prediction = estimator.predict(test_x)
        comparison_predictions[(task, name)] = np.asarray(prediction)
        valid = comparison_cv[(comparison_cv["task"] == task) &
                              (comparison_cv["model"] == name)]
        if name == "Ridge": valid = valid[valid["alpha"] == penalty]
        valid = valid.iloc[0]
        comparison_rows.append({
            "task": task, "model": name, "alpha": penalty,
            "test_r2": r2_score(test_y, prediction),
            "test_mae": mean_absolute_error(test_y, prediction),
            "test_rmse": np.sqrt(mean_squared_error(test_y, prediction)),
            "cv_r2": valid["cv_r2"], "cv_mae": valid["cv_mae"],
            "cv_rmse": valid["cv_rmse"]})
model_comparison = pd.DataFrame(comparison_rows)
model_comparison.to_csv("analysis_outputs/model_comparison.csv", index=False)
print("Training CV selects Ridge alpha:", selected_ridge_alphas)
display(model_comparison.round(4))
print("Ridge CV values were used for selection; they are not fresh independent estimates.")
print("The original test outcomes had been examined in earlier drafts.")


# Residual = recorded outcome minus prediction; positive means underprediction.
diagnostic_rows = []
fig, axes = plt.subplots(1, 2, figsize=(10, 3.1))
for task, actual, ax, title in [("2.1", np.asarray(y_test), axes[0], "2.1 Goal difference"),
                                ("2.2", np.asarray(y_test22), axes[1], "2.2 Team goals")]:
    for name, colour, marker in [("OLS", "0.55", "o"), ("Ridge", "0.1", "^")]:
        predicted = comparison_predictions[(task, name)]
        residual = actual - predicted
        diagnostic_rows.append({"task": task, "model": name,
            "mean_residual": residual.mean(), "mae": np.abs(residual).mean(),
            "rmse": np.sqrt(np.mean(residual ** 2)),
            "max_absolute_error": np.abs(residual).max(),
            "negative_predictions": int((predicted < 0).sum())})
        ax.scatter(predicted, residual, color=colour, marker=marker,
                   alpha=0.75, s=24, label=name)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Predicted outcome", ylabel="Actual minus predicted", title=title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
fig.tight_layout()
fig.savefig("analysis_outputs/Report_Test_Residuals.png", dpi=220)
plt.show()
diagnostic_summary = pd.DataFrame(diagnostic_rows)
diagnostic_summary.to_csv("analysis_outputs/diagnostic_summary.csv", index=False)
display(diagnostic_summary.round(4))
