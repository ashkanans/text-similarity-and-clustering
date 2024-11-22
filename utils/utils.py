import datetime
import random

import pandas as pd

from text_similarity.shingling import Shingling
from text_similarity.text_processing.text_preprocessor import TextPreprocessor

# List of user agents to randomize requests and avoid detection
USER_AGENTS = [
    # Windows User Agents
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",

    # macOS User Agents
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36",

    # Linux User Agents
    # "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    # "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
    # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
    # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
]


def get_random_user_agent():
    """
    Selects a random user agent from the predefined list.

    :return: str - A user agent string.
    """
    return random.choice(USER_AGENTS)


def get_headers():
    """
    Generates headers with a random user agent and additional fields.

    :return: dict - HTTP headers for a request.
    """
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.amazon.it/",
    }
    return headers


def save_results(args, shingle_comparison):
    """
    Saves similarity analysis results to a CSV file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_with_timestamp = f"{args.output.rsplit('.', 1)[0]}_{timestamp}.csv"
    print(f"Saving results to {output_file_with_timestamp}...")

    args_dict = vars(args)
    args_df = pd.DataFrame([args_dict])
    combined_df = pd.concat([args_df, shingle_comparison.df], ignore_index=True)
    combined_df.to_csv(output_file_with_timestamp, index=False)


def generate_shingles(descriptions, k, tokenized):
    """
    Generate shingles from raw or tokenized descriptions.

    Args:
        descriptions (list): List of raw text descriptions or pre-tokenized descriptions.
        k (int): Shingle length.
        tokenized (bool): Whether the input is tokenized or raw.

    Returns:
        list: List of shingle sets for each description.
    """
    shingling = Shingling(documents=descriptions, k=k, is_tokenized=tokenized)
    return shingling.shingles


def preprocess_descriptions(df, tokenize=False, filter_descriptions=False):
    """
        Preprocesses product descriptions from df DataFrame

        :param tokenize: requests preprocessing of data (tokenized) or return raw description text
        :param df: DataFrame
        :param filter_descriptions: (bool) If True, removes duplicate descriptions based on their text.
        :return: list of unique processed descriptions (tokenized lists).
    """
    text_preprocessor = TextPreprocessor(tokenize=tokenize)
    processed_descriptions = []
    unique_descriptions = set()

    if df is not None:
        for description in df['Description']:
            processed_tokens = text_preprocessor.preprocess_text(description)
            description_tuple = tuple(processed_tokens)

            if not filter_descriptions or description_tuple not in unique_descriptions:
                processed_descriptions.append(processed_tokens)
                unique_descriptions.add(description_tuple)
    else:
        print("No data to process. Please scrape or load data first.")

    print(f"Processed {len(processed_descriptions)} unique descriptions.")
    return processed_descriptions
