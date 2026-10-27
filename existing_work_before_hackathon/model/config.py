import os
_basedir = os.path.abspath(os.path.dirname(__file__))
version = 'v1'
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
SERVER_ENDPOINT = os.getenv("SERVER_ENDPOINT", "https://localhost:5173")

USER_DATA_BUCKET_NAME = os.getenv("USER_DATA_BUCKET_NAME", "aura-user-data")
HISTORY_IMAGE_BUCKET_NAME = os.getenv("HISTORY_IMAGE_BUCKET_NAME", "aura-history-image")

del os
