import ssl
import urllib.request
import nltk
import time
import csv
from bs4 import BeautifulSoup

# --- FIX FOR MACOS SSL CERTIFICATE VERIFICATION ERROR ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --------------------------------------------------------

# Ensure required NLTK data models are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_tab', quiet=True)

URL = "https://en.wikipedia.org/wiki/Characters_of_Final_Fantasy_X_and_X-2"


def fetch_and_extract_body_text(url):
    """Fetches HTML from the URL and retrieves cleaned text from the body tag."""
    print("Fetching web page...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        response = urllib.request.urlopen(req)
        html = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to fetch page: {e}")
        return ""

    print("Parsing HTML...")
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.find('body')
    if not body:
        return ""

    # Remove script, style, citation, table, and navigation elements to clean text
    for element in body(["script", "style", "sup", "table", "nav"]):
        element.extract()

    return body.get_text(separator=' ')


def split_into_statements(text):
    """Splits full text into individual sentence statements without using regex."""
    sentences = nltk.sent_tokenize(text)
    # Clean extra whitespace using standard string methods and filter short fragments
    cleaned_sentences = [" ".join(s.split()) for s in sentences if len(s.strip()) > 15]
    return cleaned_sentences


def extract_primary_entity(sentence):
    """Parses a statement for Capitalized Words / Proper Nouns acting as Entities."""
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    # Identify Proper Nouns (NNP/NNPS) and Capitalized Words
    for word, tag in pos_tags:
        if tag in ['NNP', 'NNPS'] and word[0].isupper():
            # Return the first found proper noun as the primary entity for this event
            return word

    return None


def process_page_to_event_sequences(url):
    """
    Main workflow function to generate formulas.
    Formula: [Entity] knew about [Event 1] before [Event 2]
    (Matching user template: [Entity] k*** ab*** [Event] be**** [Event])
    """
    body_text = fetch_and_extract_body_text(url)
    if not body_text:
        return [], []

    print("Tokenizing statements (Events) and extracting Entities...")
    statements = split_into_statements(body_text)

    # Construct the formula sequences
    event_sequences = []

    # We iterate through to completion (len - 1 so we can always pair with the next statement)
    for i in range(len(statements) - 1):
        event_1 = statements[i]  # Statement acting as Event 1
        event_2 = statements[i + 1]  # Statement acting as Event 2

        # Extract the Entity from the first Event
        entity = extract_primary_entity(event_1)

        # If no entity is found in this statement, skip to keep the output clean
        if not entity:
            continue

        # Formulate the string according to the requested wildcards
        # k***  = knew
        # ab*** = about
        # be**** = before
        formula_str = f"[{entity}] knew about [{event_1}] before [{event_2}]"

        event_sequences.append({
            "formula": formula_str,
            "entity": entity,
            "event_1": event_1,
            "event_2": event_2
        })

    return statements, event_sequences


def save_sequences_to_csv(sequences):
    """Saves the sequences to a CSV file with an epoch timestamp in the filename."""
    epoch_time = int(time.time())
    filename = f"event_sequences_{epoch_time}.csv"

    print(f"\nSaving {len(sequences)} results to {filename}...")

    # Writing data to CSV (Can be opened natively in Excel, Numbers, etc.)
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write headers
        writer.writerow([
            "Formula Sequence",
            "Extracted Entity",
            "Event 1 (Statement 1)",
            "Event 2 (Statement 2)"
        ])

        # Write rows to completion
        for seq in sequences:
            writer.writerow([
                seq["formula"],
                seq["entity"],
                seq["event_1"],
                seq["event_2"]
            ])

    print("Save complete!")


if __name__ == "__main__":
    statements_list, sequences = process_page_to_event_sequences(URL)

    print("\n======================================")
    print(f"Total Statements Extracted: {len(statements_list)}")
    print(f"Total Formula Sequences Generated: {len(sequences)}")
    print("======================================\n")

    # Print all sequences to completion in the console
    print("--- Formulated Event Sequences ---")
    for seq in sequences:
        print(seq["formula"])

    # Save the data to local file (CSV)
    if sequences:
        save_sequences_to_csv(sequences)
