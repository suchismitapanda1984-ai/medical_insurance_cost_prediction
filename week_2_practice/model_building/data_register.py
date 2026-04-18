
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo, login
import os

# 🔹 Replace with your actual username
repo_id = "your-username/medical-insurance-cost-prediction"
repo_type = "dataset"

# 🔹 Login using GitHub Secret
login(token=os.getenv("HF_TOKEN"))

# Initialize API client
api = HfApi()

# Step 1: Check if dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Step 2: Upload data folder
api.upload_folder(
    folder_path="week_2_practice/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("✅ Data uploaded successfully!")
