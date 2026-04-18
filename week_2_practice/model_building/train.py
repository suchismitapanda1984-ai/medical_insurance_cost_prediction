
# for data manipulation
import pandas as pd
import numpy as np
import os

# model training
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# model saving
import joblib

# hugging face
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# CONFIG
DATA_REPO = "your-username/medical-insurance-cost-prediction"
MODEL_REPO = "your-username/medical-insurance-model"

# LOGIN
login(token=os.getenv("HF_TOKEN"))
api = HfApi()

# LOAD DATA (processed)
Xtrain = pd.read_csv(f"hf://datasets/{DATA_REPO}/processed/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATA_REPO}/processed/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATA_REPO}/processed/ytrain.csv").squeeze()
ytest = pd.read_csv(f"hf://datasets/{DATA_REPO}/processed/ytest.csv").squeeze()

print("Data loaded successfully")

# MODEL
xgb_model = xgb.XGBRegressor(
    random_state=42,
    objective="reg:squarederror"
)

# HYPERPARAMETERS
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# GRID SEARCH
grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

print("Best Params:", grid_search.best_params_)

# PREDICTIONS
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# EVALUATION
print("\nTraining Performance:")
print("MAE:", mean_absolute_error(ytrain, y_pred_train))
print("RMSE:", np.sqrt(mean_squared_error(ytrain, y_pred_train)))
print("R²:", r2_score(ytrain, y_pred_train))

print("\nTest Performance:")
print("MAE:", mean_absolute_error(ytest, y_pred_test))
print("RMSE:", np.sqrt(mean_squared_error(ytest, y_pred_test)))
print("R²:", r2_score(ytest, y_pred_test))

# SAVE MODEL
os.makedirs("week_2_practice/models", exist_ok=True)

model_path = "week_2_practice/models/best_model.joblib"
joblib.dump(best_model, model_path)

print("Model saved locally")

# CREATE MODEL REPO
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' exists.")
except RepositoryNotFoundError:
    print(f"Creating model repo '{MODEL_REPO}'...")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

# UPLOAD MODEL
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model",
)

print("Model uploaded to Hugging Face!")
