import logging
import os
from dotenv import load_dotenv
load_dotenv()

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)
