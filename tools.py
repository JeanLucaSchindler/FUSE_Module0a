import sqlite3
import random
import numpy as np
from sklearn.metrics import jaccard_score, hamming_loss



def select_random_keys(dictionary, n=100, seed=22):
    """
    Randomly selects `n` keys from the given dictionary.

    Args:
        dictionary (dict): The input dictionary.
        n (int): The number of keys to select. Defaults to 50.

    Returns:
        list: A list of randomly selected keys.
    """
    if len(dictionary) < n:
        raise ValueError("The dictionary contains fewer keys than the number to select.")

    # Set the seed for reproducibility
    random.seed(seed)

    return random.sample(list(dictionary.keys()), n)




def save_answers_to_sqlite(database_path, table_name, data_dict1, data_dict2):
    """
    Creates an SQLite table from two dictionaries.
    The first dictionary's keys and values are stored in columns 'Text' and 'output',
    and the values from the second dictionary (based on the first dictionary's keys)
    are stored in column 'correct_answer'.

    Args:
        database_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to be created.
        data_dict1 (dict): The first dictionary to populate 'Text' and 'output' columns.
        data_dict2 (dict): The second dictionary to populate the 'correct_answer' column.

    Returns:
        None
    """
    try:
        # Connect to the SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Create the table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                Text TEXT,
                output TEXT,
                correct_answer TEXT
            )
        """)

        # Prepare data for insertion
        data_to_insert = [
            (key, str(value).replace(']', '').replace('[', '').replace("'", ''), str(data_dict2.get(key, None)).replace("'",'').replace('{', '').replace('}', ''))  # Use None if key not found in data_dict2
            for key, value in data_dict1.items()
        ]

        # Insert data into the table
        cursor.executemany(f"""
            INSERT INTO {table_name} (Text, output, correct_answer)
            VALUES (?, ?, ?)
        """, data_to_insert)

        # Commit changes
        conn.commit()

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")

    finally:
        # Ensure the connection is always closed
        if conn:
            conn.close()


def create_binary_matrix(themes, words_list):
    """
    Create a binary matrix for multilabel classification from a list of words.

    Parameters:
    themes (dict): A dictionary where keys are theme codes and values are theme names.
    words_list (list of list): Each sublist contains words corresponding to the labels for one sample.

    Returns:
    np.ndarray: A binary label matrix of shape (n_samples, 33).
    """
    theme_names = list(themes.values())
    binary_matrix = []

    for words in words_list:
        binary_row = [1 if theme in words else 0 for theme in theme_names]
        binary_matrix.append(binary_row)

    # Ensure the matrix has 33 columns
    binary_matrix = np.array(binary_matrix)
    assert binary_matrix.shape[1] == 33, "Binary matrix must have exactly 33 columns."

    return binary_matrix



def evaluate_metrics(y_true, y_pred):
    """
    Evaluate Jaccard Index and Hamming Loss for multilabel classification.

    Parameters:
    y_true (np.ndarray): Ground truth binary label matrix (shape: n_samples x n_classes).
    y_pred (np.ndarray): Predicted binary label matrix (shape: n_samples x n_classes).

    Returns:
    dict: A dictionary containing Jaccard Index (average) and Hamming Loss.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Compute Jaccard Index (macro-average)
    jaccard_avg = jaccard_score(y_true, y_pred, average='samples')

    # Compute Hamming Loss
    hamming = hamming_loss(y_true, y_pred)

    return jaccard_avg, hamming

import numpy as np
from scipy.stats import t

def personalized_metric(y_true, y_pred):
    """
    Calculate a personalized metric for multilabel classification.

    Parameters:
    y_true (np.ndarray): Ground truth binary label matrix (shape: n_samples x n_classes).
    y_pred (np.ndarray): Predicted binary label matrix (shape: n_samples x n_classes).

    Returns:
    tuple: mean_score, std_dev, (lower_95, upper_95), scores
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    scores = []

    for true_row, pred_row in zip(y_true, y_pred):

        # calculate percentage of correctly predicted themes
        correct_themes = np.sum((true_row == 1) & (pred_row == 1))
        total_themes = np.sum(true_row)
        score = 100 * (correct_themes / total_themes)

        # Count incorrect predicted themes
        incorrect_themes = np.sum((true_row == 0) & (pred_row == 1))
        score -= incorrect_themes * 5

        scores.append(score)

    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_dev = np.std(scores, ddof=1)
    n = len(scores)

    # Calculate the 95% confidence interval
    t_critical = t.ppf(0.975, df=n-1)  # Two-tailed critical value for 95% CI
    margin_of_error = t_critical * (std_dev / np.sqrt(n))
    lower_95 = mean_score - margin_of_error
    upper_95 = mean_score + margin_of_error

    return mean_score, std_dev, (lower_95, upper_95), scores



def create_metrics_db(db_name):
    """
    Create a SQLite database and a table with the specified columns if they don't exist.
    """
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    # Create table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts (
            prompt_name TEXT,
            prompt TEXT,
            jaccard_index REAL,
            hamming_loss REAL,
            personalized_loss REAL
        )
        """
    )

    connection.commit()
    connection.close()

def prompt_metrics_save(prompt_name, prompt, jaccard_index, hamming_loss, personalized_loss):
    """
    Add a row to the SQLite database.
    """
    create_metrics_db('data/prompt_metrics.db')

    connection = sqlite3.connect('data/prompt_metrics.db')
    cursor = connection.cursor()

    # Insert a new row
    cursor.execute(
        """
        INSERT INTO prompts (prompt_name, prompt, jaccard_index, hamming_loss, personalized_loss)
        VALUES (?, ?, ?, ?, ?)
        """,
        (prompt_name, prompt, jaccard_index, hamming_loss, personalized_loss)
    )

    connection.commit()
    connection.close()
