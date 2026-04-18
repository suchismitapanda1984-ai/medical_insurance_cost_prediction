
# for data manipulation
import pandas as pd
import os

# for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# hugging face
from huggingface_hub import login, HfApi

# CONFIG
REPO_ID = "your-username/medical-insurance-cost-prediction"
DATASET_PATH = f"hf://datasets/{REPO_ID}/insurance.csv"

# LOGIN
login(token=os.getenv("HF_TOKEN"))
api = HfApi()

# LOAD DATA
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# FEATURES
target_col = "charges"

X = df.drop(columns=[target_col])
y = df[target_col]

# COLUMN TYPES
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

# PREPROCESSOR (CORRECT WAY)
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), cat_cols)
], remainder="passthrough")

X_processed = preprocessor.fit_transform(X)

# Convert back to DataFrame
X_processed = pd.DataFrame(X_processed)

# TRAIN TEST SPLIT
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# CREATE OUTPUT FOLDER
os.makedirs("week_2_practice/data/processed", exist_ok=True)

# SAVE FILES
Xtrain.to_csv("week_2_practice/data/processed/Xtrain.csv", index=False)
Xtest.to_csv("week_2_practice/data/processed/Xtest.csv", index=False)
ytrain.to_csv("week_2_practice/data/processed/ytrain.csv", index=False)
ytest.to_csv("week_2_practice/data/processed/ytest.csv", index=False)

print("Data split and saved locally.")

# UPLOAD FILES
files = [
    "week_2_practice/data/processed/Xtrain.csv",
    "week_2_practice/data/processed/Xtest.csv",
    "week_2_practice/data/processed/ytrain.csv",
    "week_2_practice/data/processed/ytest.csv",
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed/{os.path.basename(file_path)}",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

print("Processed data uploaded to Hugging Face.")
