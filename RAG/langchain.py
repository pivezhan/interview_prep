#%%
import os
import datetime
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Load .env and validate key
_ = load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

# Initialize the client (env var is used automatically, passing is fine too)
client = OpenAI(api_key=api_key)

#%%
# Model selection (update away from deprecated 3.5-turbo)
# See: https://platform.openai.com/docs/models
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)

# Prefer a current lightweight model; adjust as you like.
# You can also use 'gpt-5' if enabled for your account via the Responses API.
if current_date > target_date:
    llm_model = "gpt-4o-mini"
else:
    llm_model = "gpt-4o-mini"  # keep same to avoid deprecated 3.5 variants

#%%
def get_completion(prompt: str, model: str = None, temperature: float = 0):
    """
    Minimal wrapper around Chat Completions in the new SDK.
    """
    model = model or llm_model
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    # New SDK returns .choices[i].message.content
    return resp.choices[0].message.content

#%%
# Example
print(get_completion("what is 1+1?"))


