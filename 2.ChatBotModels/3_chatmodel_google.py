from langchain_google_genai import ChatGoogleGenrativeAI
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenrativeAI(model="gemini-1.5-pro", temperature=0.9, max_comletion_tokens=10)
result = model.invoke("What is the capital of India?")
print(result)
print(type(result.content))