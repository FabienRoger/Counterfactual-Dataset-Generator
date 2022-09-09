import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
VERBOSE = int(os.environ.get("CDG_VERBOSE", "0"))
