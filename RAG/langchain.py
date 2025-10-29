#%%
!pip install python-dotenv
!pip install openai
#%%
#%%
import os
from openai import OpenAI  # Import the new client

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# Optional: Debug print (remove in production)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

# Initialize the client with your API key
client = OpenAI(api_key=api_key)
#%%
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

#%%
def get_completion(prompt, model="gpt-3.5-turbo"):
	messages = [{"role": "user", "content": prompt}]
	response = OpenAI.ChatCompletion.create(
		model=model,
		messages=messages,
		temperature=0, # this is the degree of randomness of the model's output
	)
	return response.choices[0].message["content"]

#%%
get_completion("what is 1+1?")

#%%
