from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import json
import requests
import time

# Load environment variables from .env file
load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-4o-search-preview",
    web_search_options={},
    messages=[
        {
            "role": "user",
            "content": "what is the price of the item {item_name} in {location}",
        }
    ],
)