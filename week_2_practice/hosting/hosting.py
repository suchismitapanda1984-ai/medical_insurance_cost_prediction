
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# CONFIG
REPO_ID = "your-username/medical-insurance-cost-prediction"

# LOGIN
login(token=os.getenv("HF_TOKEN"))
api = HfApi()

# CHECK / CREATE SPACE
try:
    api.repo_info(repo_id=REPO_ID, repo_type="space")
    print(f"Space '{REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating Space '{REPO_ID}'...")
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False
    )

# UPLOAD FILES
api.upload_folder(
    folder_path="week_2_practice/deployment",
    repo_id=REPO_ID,
    repo_type="space",
    path_in_repo=""
)

print("Deployment pushed to Hugging Face Space successfully!")
