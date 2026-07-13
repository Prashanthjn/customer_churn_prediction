import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Debug (optional)
print("AWS_ACCESS_KEY:", AWS_ACCESS_KEY)
print("AWS_REGION:", AWS_REGION)
print("BUCKET_NAME:", BUCKET_NAME)