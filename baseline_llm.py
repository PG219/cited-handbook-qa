from google import genai
import os

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key = "AIzaSyB5cV6AoWMRs8ZQPitqlxIjOAqpxCpMQSU")
def load_text(directory):
    files = sorted(os.listdir(directory))
    full_text = ""
    for filename in files:
        with open(os.path.join(directory, filename), 'r', encoding = 'utf-8') as f:
            full_text += f"\n\n--- DOCUMENT: {filename} ---\n" + f.read()
    return full_text
context = load_text("./data")
question = "Who is Pranay"
prompt = f"""Use the following documentation to answer the question. 
If the answer is not in the text, say 'I do not know'.
CONTEXT:
{context}

QUESTION:
{question}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-lite", # Or "gemini-3-flash-preview" if available
    contents=prompt
)
print(f"LLM RESPONSE: {response.text}")