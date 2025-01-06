import PyPDF2
from dotenv import load_dotenv
import openai
import os
# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')



def extract_text_from_pdf(pdf_path):
    pdf_text = {}
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            cleaned_text = ' '.join(text.split())
            pdf_text[page_num + 1] = cleaned_text
    return pdf_text


def get_theme_allocation(page_text, system_instructions):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
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


def replace_themes_p1(lists, themes):
    replaced_lists = []
    for lst in lists:
        # Extract entries from the list and replace themes
        entries = [item.strip() for item in lst[1:-1].split(",")]
        replaced_entries = [themes.get(entry, entry) for entry in entries]
        # Format back into list structure
        replaced_lists.append("[" + ", ".join(replaced_entries) + "]")
    return replaced_lists

def replace_themes_p2(lists, themes):
    replaced_lists = []
    for lst in lists:
        # Extract entries from the list and replace themes
        entries = [item.strip() for item in lst[1:-1].split(",")]
        replaced_entries = [
            f"{themes.get(entry.split(':')[0], entry.split(':')[0])}:{entry.split(':')[1]}" for entry in entries
        ]
        # Format back into list structure
        replaced_lists.append("[" + ", ".join(replaced_entries) + "]")
    return replaced_lists

# Function to sort each list by C1, C2, C3
def sort_list_by_column(data):
    # Extract tuples (Tn, Cx)
    entries = [item.strip() for item in data[1:-1].split(",")]
    parsed = [(item.split(":")[0], item.split(":")[1]) for item in entries]
    # Sort by column (C1, C2, C3)
    sorted_entries = sorted(parsed, key=lambda x: x[1], reverse=True)
    # Format back to the original structure
    return "[" + ", ".join([f"{key}:{value}" for key, value in sorted_entries]) + "]"
