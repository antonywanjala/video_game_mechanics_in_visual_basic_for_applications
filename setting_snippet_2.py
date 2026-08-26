from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm
import ssl
import time
import pandas as pd

# 1. Bypass SSL certificate verification
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


def extract_data_from_url(url):
    # 3. Request the web page using Selenium to bypass Cloudflare
    print(f"Fetching content from: {url} using Selenium (Firefox)...")

    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.set_preference("general.useragent.override",
                                   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0")

    try:
        driver = webdriver.Firefox(options=firefox_options)
        driver.get(url)
        time.sleep(3)
        html_content = driver.page_source
    except Exception as e:
        print(f"Failed to retrieve the webpage. Error: {e}")
        return [], []
    finally:
        try:
            driver.quit()
        except:
            pass

    # 4. Parse the HTML and import all text in the body into a local variable
    print("Parsing HTML...")
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    if soup.body:
        local_body_text = " ".join(soup.body.get_text(separator=' ', strip=True).split())
    else:
        local_body_text = " ".join(soup.get_text(separator=' ', strip=True).split())

    # 5. Differentiate each statement into a list
    print("Tokenizing sentences...")
    statements = nltk.sent_tokenize(local_body_text)

    # This will hold the dictionary for each differentiated statement
    statements_data = []

    # 6. Parse each statement for capitalized words, proper nouns, and entities
    for index, statement in enumerate(tqdm(statements, desc="Extracting entities and actions", unit="stmt")):
        words = nltk.word_tokenize(statement)
        tagged_words = nltk.pos_tag(words)

        # Extract potential characters (Proper Nouns) and actions (Verbs)
        characters = [word for word, pos in tagged_words if pos in ('NNP', 'NNPS')]
        actions = [word for word, pos in tagged_words if pos.startswith('VB')]

        character = characters[0] if characters else None
        action = actions[0] if actions else None
        event = None

        if character and action:
            event = f"{character} {action}"

        # Store the differentiated statement in a dictionary
        statements_data.append({
            "Statement ID": index + 1,
            "Raw Statement": statement,
            "Character (Proper Noun)": character,
            "Action (Verb)": action,
            "Formed Event": event
        })

    # 7. Feed the entities and statements into the formula:
    # [Entity] knew about [Event and/or Entity] before [Event and/or Entity] (where statements are Events)
    event_chain = []

    # Filter for statements that successfully extracted at least a Character (Entity) to use in the formula
    valid_statements = [data for data in statements_data if data["Character (Proper Noun)"]]

    for i in tqdm(range(len(valid_statements) - 1), desc="Formatting event chain", unit="formula"):
        entity = valid_statements[i]["Character (Proper Noun)"]

        # "where statements are Events" -> mapping the raw statement to the Event blocks
        event1 = valid_statements[i]["Raw Statement"]
        event2 = valid_statements[i + 1]["Raw Statement"]

        # Apply the new requested formula
        formatted_sequence = f"[{entity}] knew about [{event1}] before [{event2}]"
        event_chain.append(formatted_sequence)

    return statements_data, event_chain


# --- Example Usage ---
if __name__ == "__main__":
    target_url = "https://finalfantasy.fandom.com/wiki/Final_Fantasy_X_timeline"

    statements_dict_list, events_output = extract_data_from_url(target_url)

    print(f"\nExtracted {len(statements_dict_list)} statements and {len(events_output)} event sequences.\n")

    if statements_dict_list:
        # Generate the current epoch time for the filenames
        epoch_time = int(time.time())

        # --- Save Statements Dictionary List ---
        statements_csv = f"differentiated_statements_{epoch_time}.csv"
        statements_xlsx = f"differentiated_statements_{epoch_time}.xlsx"

        df_statements = pd.DataFrame(statements_dict_list)
        df_statements.to_csv(statements_csv, index=False, encoding='utf-8')
        print(f"Saved statements to CSV: {statements_csv}")

        # --- Save Event Sequence List ---
        sequences_csv = f"event_sequences_knew_about_{epoch_time}.csv"
        sequences_xlsx = f"event_sequences_knew_about_{epoch_time}.xlsx"

        df_sequences = pd.DataFrame(events_output, columns=["Knowledge Sequence Formula"])
        df_sequences.to_csv(sequences_csv, index=False, encoding='utf-8')
        print(f"Saved event sequences to CSV: {sequences_csv}")

        # Save both to XLSX
        try:
            df_statements.to_excel(statements_xlsx, index=False)
            df_sequences.to_excel(sequences_xlsx, index=False)
            print(f"Saved statements to XLSX: {statements_xlsx}")
            print(f"Saved event sequences to XLSX: {sequences_xlsx}\n")
        except ModuleNotFoundError:
            print(f"Could not save to XLSX. Please install 'openpyxl' (pip install openpyxl).\n")

        # Print the complete event sequence interval
        print("--- Complete Event Sequence Formula ---")
        for e in events_output:
            print(e)
    else:
        print("No data was extracted to save.")
