import pandas as pd

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
