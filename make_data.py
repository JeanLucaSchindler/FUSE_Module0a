import os
import glob
import unicodedata
from docx import Document
from rapidfuzz import fuzz
import pandas as pd
import json

# Translations from French to English

translations = {
    'Amour': 'Love',
    'Art': 'Art',
    'Beauté': 'Beauty',
    'Bien': 'Goodness',
    'Conscience': 'Conscience',
    'Corps': 'Body & Mind',
    'Désir': 'Desire',
    'Education': 'Education',
    'Existence': 'Existence',
    'Folie': 'Madness',
    'Histoire': 'History',
    'Identité': 'Identity',
    'Imagination': 'Imagination',
    'Inconscient': 'The Unconscious',
    'Joie': 'Joy & Happiness',
    'Justice': 'Justice',
    'Langage': 'Language',
    'Liberté': 'Freedom',
    'Matière': 'Matter',
    'Mort': 'Death',
    'Penser': 'Thought',
    'Philosophie': 'Philosophy',
    'Politique': 'Politics',
    'Reel': 'Reality',
    'Religion': 'Religion',
    'Science': 'Science',
    'Sens': 'Meaning',
    'Technique': 'Technology',
    'Temps': 'Time',
    'Travail': 'Work',
    'Vérité': 'Truth',
    'Vivre ensemble': 'Living Together'
}

# Get dict of themes and questions

def clean_question(question):
    if pd.isna(question):
        return ''
    return question.replace('\xa0?\xa0', '').replace(' ?', '').replace('?', '').replace('\r\n', ' ')


def get_theme(theme_dataframe):
    """
    return dictionary with question as key and theme associated as value
    """

    theme_dict = {}
    for index, row in theme_dataframe.iterrows():

        question = clean_question(row['QUESTION'])
        other_question = row['AUTRE FORMULATION']
        theme = row['THEME']

        theme_dict[question] = translations[theme]

        if not pd.isna(other_question):
            other_question = clean_question(other_question)
            theme_dict[other_question] = translations[theme]


    return theme_dict




# create database in json format


def normalize_text(text):
    """
    1) Convert accented characters to their unaccented base (e.g., é -> e).
    2) Convert to lowercase.
    3) Remove commas (',') and hyphens ('-').
    """
    text_no_accents = ''.join(
        ch for ch in unicodedata.normalize('NFKD', text)
        if unicodedata.category(ch) != 'Mn'
    )
    text_no_accents = text_no_accents.lower()
    text_no_accents = text_no_accents.replace(',', '').replace('-', '')
    return text_no_accents

def is_fuzzy_match(paragraph_text, target_text, threshold=85):
    """
    Returns True if 'paragraph_text' is considered a match for 'target_text'
    above a given similarity threshold (0-100).

    Uses RapidFuzz's fuzz.ratio for approximate matching.
    """
    score = fuzz.ratio(paragraph_text, target_text)
    return score >= threshold

def paragraph_matched_categories_fuzzy(paragraph_text, sentence_category_dict, threshold=85):
    """
    Returns a set of categories for all fuzzy-matched sentences in 'paragraph_text'.

    'sentence_category_dict' is a dict like:
        {
          "Le coeur a ses raisons que la raison ignore": "Love",
          "L'art peut-il manifester la vérité": "Art",
          ...
        }

    We do a naive 'sliding window' approach to detect fuzzy matches.
    If multiple matched sentences share the same category, we store that category once.
    """
    matched_categories = set()
    norm_para = normalize_text(paragraph_text)

    for sentence, category in sentence_category_dict.items():
        norm_sentence = normalize_text(sentence)
        if not norm_sentence:
            continue
        length_s = len(norm_sentence)

        # Slide over the paragraph in windows of length(len(norm_sentence))
        for start_idx in range(len(norm_para) - length_s + 1):
            chunk = norm_para[start_idx:start_idx + length_s]
            if is_fuzzy_match(chunk, norm_sentence, threshold=threshold):
                matched_categories.add(category)
                break  # no need to search further for this sentence

    return matched_categories

def chunk_docx_paragraphs_fuzzy(
    file_path,
    sentence_category_dict,
    threshold=85
):
    """
    EXACT SAME LOGIC for chunk splitting:

    1. Read paragraphs of the .docx file.
    2. For each paragraph, check if it has a fuzzy match (any category).
    3. Split (finalize a chunk) if:
         (current paragraph has match) AND (the next paragraph does NOT).
    4. Exclude matched paragraphs from the chunk text.

    Returns a dict {chunk_text: categories_set}.

    NOTE:
    - If the last paragraph doesn't meet 'current_has_key and not next_has_key',
      it will NOT produce a final chunk (matching the original code).
    """
    try:
        doc = Document(file_path)
        paragraphs = doc.paragraphs

        chunks_dict = {}
        current_chunk_paragraphs = []
        current_chunk_categories = set()

        for index, paragraph in enumerate(paragraphs):
            para_text = paragraph.text

            matched_categories_here = paragraph_matched_categories_fuzzy(
                para_text,
                sentence_category_dict,
                threshold=threshold
            )
            current_has_key = (len(matched_categories_here) > 0)

            # Add these categories to the running chunk categories
            current_chunk_categories |= matched_categories_here

            # If this paragraph is matched, we exclude it from the chunk text
            if not current_has_key:
                current_chunk_paragraphs.append(para_text)

            # Check the next paragraph
            if index < len(paragraphs) - 1:
                next_text = paragraphs[index + 1].text
                next_matched_categories = paragraph_matched_categories_fuzzy(
                    next_text,
                    sentence_category_dict,
                    threshold=threshold
                )
                next_has_key = (len(next_matched_categories) > 0)
            else:
                # No next paragraph
                next_has_key = False

            # Split condition
            if current_has_key and not next_has_key:
                chunk_text = "\n".join(current_chunk_paragraphs).strip()
                chunks_dict[chunk_text] = current_chunk_categories

                # Reset for next chunk
                current_chunk_paragraphs = []
                current_chunk_categories = set()

        return chunks_dict

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return {}

def chunk_all_docx_in_folder(folder_path, sentence_category_dict, threshold=85):
    """
    Applies the EXACT same chunking logic (chunk_docx_paragraphs_fuzzy) to every
    .docx file in 'folder_path' and merges results into a single dictionary.

    If the same 'chunk_text' appears in multiple files, their categories are unified
    (set union).

    Returns a dict {chunk_text: set_of_categories}.
    """
    # Collect all .docx files in the folder
    docx_files = glob.glob(os.path.join(folder_path, '*.docx'))
    big_dict = {}

    for file_path in docx_files:
        partial_dict = chunk_docx_paragraphs_fuzzy(file_path, sentence_category_dict, threshold)

        # Merge partial_dict into big_dict
        for chunk_text, categories_set in partial_dict.items():
            if chunk_text in big_dict:
                big_dict[chunk_text].update(categories_set)
            else:
                big_dict[chunk_text] = set(categories_set)

    return big_dict




# save and load database


def save_dict_to_json(data_dict, file_path):
    # Convert sets to lists so JSON can handle them
    serializable_dict = {key: list(value) for key, value in data_dict.items()}

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, ensure_ascii=False, indent=2)

def load_dict_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Convert lists back to sets
    return {key: set(value) for key, value in data.items()}




if __name__ == "__main__":

    themes = pd.read_csv('data/themes.csv')
    themes = themes.drop_duplicates(subset='QUESTION')
    themes_dict = get_theme(themes)

    folder_path = "data/Extraits"
    threshold = 85

    merged_dict = chunk_all_docx_in_folder(folder_path, themes_dict, threshold=threshold)

    save_path = "data/M0A_train_data.json"

    save_dict_to_json(merged_dict, save_path)
