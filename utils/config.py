# utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CDC_URLS = [
    "https://www.cdc.gov/flu/symptoms/index.html",
    "https://www.cdc.gov/cancer/breast/basic_info/index.htm",
]

