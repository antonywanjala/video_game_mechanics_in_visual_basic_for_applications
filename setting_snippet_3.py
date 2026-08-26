from installmissingpackages12 import preset1

preset1()
import requests
import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm
import ssl
import time
import pandas as pd

# 1. Bypass SSL certificate verification for macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Download necessary NLTK models
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)


def extract_events_from_url(url):
    print(f"Fetching content from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return []

    print("Parsing HTML...")
    soup = BeautifulSoup(response.text, 'html.parser')

    if soup.body:
        body_text = soup.body.get_text(separator=' ', strip=True)
    else:
        body_text = soup.get_text(separator=' ', strip=True)

    print("Tokenizing sentences...")
    statements = nltk.sent_tokenize(body_text)

    # Store tuples of (Entity, Statement)
    extracted_data = []

    for statement in tqdm(statements, desc="Extracting entities", unit="stmt"):
        words = nltk.word_tokenize(statement)
        tagged_words = nltk.pos_tag(words)

        # Extract potential characters/entities (Proper Nouns)
        characters = [word for word, pos in tagged_words if pos in ('NNP', 'NNPS')]

        # If an entity is found, associate it with the statement (which acts as the event)
        if characters:
            entity = characters[0]
            extracted_data.append((entity, statement))

    event_chain = []

    # Apply the new formula: [Entity] knew about [Event 1] before [Event 2]
    for i in tqdm(range(len(extracted_data) - 1), desc="Formatting new event formula", unit="formula"):
        entity = extracted_data[i][0]
        event_1 = extracted_data[i][1]  # The current statement
        event_2 = extracted_data[i + 1][1]  # The following statement

        formatted_sequence = f"[{entity}] knew about [{event_1}] before [{event_2}]"
        event_chain.append(formatted_sequence)

    return event_chain


# --- Example Usage ---
if __name__ == "__main__":
    target_url = "https://en.wikipedia.org/wiki/Final_Fantasy_X"

    events_output = extract_events_from_url(target_url)

    print(f"\nExtracted {len(events_output)} formatted sequences.\n")

    if events_output:
        # Generate the epoch timestamp for unique filenames
        epoch_time = int(time.time())
        csv_filename = f"knew_about_events_{epoch_time}.csv"
        xlsx_filename = f"knew_about_events_{epoch_time}.xlsx"

        print(f"Saving results to {csv_filename} and {xlsx_filename}...")

        # Load the events into a Pandas DataFrame
        df = pd.DataFrame(events_output, columns=["Formatted_Sequence"])

        # Save to CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8')

        # Save to XLSX
        try:
            df.to_excel(xlsx_filename, index=False, engine='openpyxl')
        except ImportError:
            print(
                "⚠️ Warning: 'openpyxl' is required to save as .xlsx. Run 'pip install openpyxl'. Skipping Excel save.")

        print("Save complete!\n")

    # Print ALL event sequences to completion
    print("--- FULL FORMATTED LOG ---")
    for e in events_output:
        print(e)
