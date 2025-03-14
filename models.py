from dotenv import load_dotenv
import os
from google import genai
from openai import OpenAI

# Load environment variables
load_dotenv()
deepseek_api = os.getenv('DEEPSEEK_API_KEY')
gemini_api = os.getenv('GEMINI_API_KEY')

clientgpt = OpenAI(api_key=os.environ.get("FUSE_OPEN_AI_KEY"))
clientgem = genai.Client(api_key=gemini_api)
client_deepseek = OpenAI(api_key=deepseek_api, base_url="https://api.deepseek.com")


def get_theme_allocation_deepseek(page_text, system_instructions, model="deepseek-reasoner"):
    try:
        response = client_deepseek.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": page_text},],stream=False)

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error processing page: {str(e)}")

def get_theme_allocation_gemini(page_text, system_instructions, model="gemini-2.0-flash"):
    try:
        response = clientgem.models.generate_content(
            model=model, contents=f"{system_instructions}. Text: {page_text}"
        )
        return response.text.strip()

    except Exception as e:
        print(f"Error processing page: {str(e)}")

def get_theme_allocation_openai(page_text, system_instructions, model):
    try:
        response = clientgpt.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": page_text}
            ],
            max_tokens=100,
            temperature=0.0
        )
        allocation = response.choices[0].message.content.strip()
        return allocation

    except Exception as e:
        print(f"Error processing page: {str(e)}")
