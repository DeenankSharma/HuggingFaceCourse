import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")
