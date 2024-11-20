import os
import random
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urlunparse, urljoin

import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup

from text_similarity.text_processing.TextPreprocessor import TextPreprocessor


class AmazonScraper:
    def __init__(self, keywords, num_pages):
        self.scraped_results = None
        self.df = None
        self.keywords = keywords
        self.num_pages = num_pages
        self.base_url = "https://www.amazon.it/s"
        self.user_agents = [
            # Windows User Agents
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",

            # # macOS User Agents
            # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
            # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            # "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
            # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15",
            # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36",
            #
            # # Linux User Agents
            # "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
            # "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
            # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
            # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
        ]

        self.data = []
        self.text_preprocessor = TextPreprocessor()
        self.keyword = self.keywords.split(",")[0]
        self.params = {"k": self.keyword, "page": self.num_pages}

    def get_random_user_agent(self):
        # Select a random user agent from the list
        return random.choice(self.user_agents)

    def get_headers(self):
        # Set headers with a random user agent and additional fields
        headers = {
            "User-Agent": self.get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.amazon.it/",
        }
        return headers

    def scrape_amazon_products(self):
        data = []

        for keyword in self.keywords.split(","):
            keyword = keyword.strip()
            print(f"Scraping keyword {keyword}...")
            for page in range(1, self.num_pages + 1):
                url = f"{self.base_url}?k={keyword}&page={page}"
                headers = self.get_headers()
                response = requests.get(url, params=self.params, headers=headers)
                time.sleep(5)

                if response.status_code == 200:
                    print(f"Scraping page {page}...")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    products = soup.find_all('div', class_='s-result-item')

                    for product in products:
                        try:
                            description_element = product.find('span',
                                                               {'class': 'a-size-base-plus a-color-base a-text-normal'})
                            product_description = description_element.text.strip()
                            if "'" in product_description:
                                product_description = product_description.replace("'", "")
                            elif '"' in product_description:
                                product_description = product_description.replace('"', '')
                        except AttributeError:
                            continue

                        try:
                            anchor_parent = product.find('a', {'class': 'a-link-normal s-no-outline'})

                            if anchor_parent is not None:
                                product_url = anchor_parent.get('href')
                                product_url = urljoin('https://www.amazon.it/', product_url)
                            else:
                                continue
                        except (AttributeError, TypeError):
                            continue

                        try:
                            price_element = product.find('span', {'class': 'a-price-whole'})
                            product_price = float(price_element.text.strip().replace('.', '').replace(',', '.'))
                            if product_price == 11999:
                                product_price = 0.0
                        except AttributeError:
                            product_price = 0.0

                        try:
                            prime_element = product.find('i', {'class': 'a-icon a-icon-prime a-icon-medium',
                                                               'aria-label': 'Amazon Prime'})
                            prime_product = True if prime_element is not None else False
                        except AttributeError:
                            prime_product = None

                        try:
                            star_element = product.find('i',
                                                        {
                                                            'class': 'a-icon a-icon-star-small a-star-small-4-5 aok-align-bottom'})
                            star_rating_text = star_element.find('span', {'class': 'a-icon-alt'}).text
                            star_rating_parts = star_rating_text.split()
                            if len(star_rating_parts) >= 1:
                                star_rating = float(star_rating_parts[0].replace(',', '.'))
                            else:
                                star_rating = 0.0
                        except (AttributeError, ValueError):
                            star_rating = 0.0

                        try:
                            review_element = product.find('span', {'class': 'a-size-base s-underline-text'})
                            raw_text = review_element.text.strip()
                            num_reviews = int(raw_text.replace('.', ''))
                        except (AttributeError, ValueError):
                            num_reviews = 0

                        data.append(
                            [product_description, product_price, prime_product, product_url, star_rating, num_reviews])
                else:
                    print(f"Error retrieving page {page}.")
                    break
            self.data = data
            print(f"{len(data)} products found")

    def save_to_tsv(self):
        # Define output folder as 'data/raw' and ensure it exists
        output_folder = os.path.join("data", "raw")
        os.makedirs(output_folder, exist_ok=True)

        # Define the file path with directory and file name, including current date
        format = "%Y-%m-%d"
        current_time = datetime.now().strftime(format)
        file_path = os.path.join(output_folder, f"{self.keyword}_results_{current_time}.tsv")

        # Define the columns and create DataFrame, removing duplicates based on 'Description'
        columns = ['Description', 'Price', 'Prime Product', 'URL', 'Stars', 'Reviews']
        df = pd.DataFrame(self.data, columns=columns)
        df = df.drop_duplicates(subset=['Description'])

        # Save the cleaned DataFrame to a .tsv file
        df.to_csv(file_path, sep='\t', index=False)
        self.scraped_results = file_path
        print("Data saved to file:", file_path)

    def load_dataset(self, file_path=None):
        # Construct the default file path in 'data/raw' if no file path is provided
        if file_path is None:
            file_path = os.path.join("data", "raw", f"{self.keyword}_results.tsv")

        # Load the dataset, dropping rows with missing values
        df = pd.read_csv(file_path, sep='\t')
        df = df.dropna()

        print("Data loaded successfully.")
        self.df = df

    def analyze_data(self, df):
        df = df.dropna()

        # 1. Price Ranges
        price_bins = [0, 100, 200, 300, 400, 500, float('inf')]
        price_labels = ['<100', '100-200', '200-300', '300-400', '400-500', '500+']
        df['Price Range'] = pd.cut(df['Price'], bins=price_bins, labels=price_labels)
        price_range_counts = df['Price Range'].value_counts()
        print("Price Ranges:")
        print(price_range_counts)

        # Visualize the distribution of prices across different categories using a box plot
        fig = px.box(df, x='Price Range', y='Price', points="all", title='Distribution of Prices Across Categories')
        fig.show()

        # 2. Customer Reviews
        top_rated_products = df.nlargest(10, 'Stars')
        self.plot_top_rated_products(top_rated_products)
        self.scatter_plot_rating_vs_reviews()
        print("\nTop Rated Products:")
        print(top_rated_products)

        # 3. Primeness
        prime_products = df[df['Prime Product'] == True]
        non_prime_products = df[df['Prime Product'] == False]
        print("\nPrime Products:")
        print(prime_products.describe())
        print("\nNon-Prime Products:")
        print(non_prime_products.describe())

        # 4. Plot the top 10 products in terms of ratings
        top_10_ratings = df.nlargest(10, 'Stars')
        self.plot_top_products(top_10_ratings, 'Stars', 'Description', 'Top 10 Products by Star Rating')

        # 5. Plot the top 10 products in terms of price
        top_10_prices = df.nlargest(10, 'Price')
        self.plot_top_products(top_10_prices, 'Price', 'Description', 'Top 10 Products by Price')

        # 6. Scatter plot of price vs. star rating
        self.scatter_plot_price_vs_rating()

    def plot_top_products(self, df, x_column, y_column, title):
        fig = px.bar(df, x=x_column, y='Description', orientation='h', title=title,
                     color='Prime Product')

        fig.update_layout(yaxis=dict(type='category'))
        fig.show()

    def scatter_plot_price_vs_rating(self):
        fig = px.scatter(self.df, x='Price', y='Stars', color='Prime Product', size='Reviews',
                         hover_data=['Description'], title='Scatter Plot: Price vs. Star Rating')

        fig.update_layout(xaxis_title='Price (EUR)', yaxis_title='Star Rating')
        fig.show()

    def plot_top_rated_products(self, top_rated_products):
        fig = px.bar(top_rated_products,
                     x='Stars',
                     y='Description',
                     color='Prime Product',
                     text='Reviews',
                     orientation='h',
                     title='Top Rated Products with Customer Reviews')

        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(yaxis=dict(type='category'))
        fig.show()

    def scatter_plot_rating_vs_reviews(self):
        fig = px.scatter(self.df, x='Stars', y='Reviews', color='Prime Product',
                         hover_data=['Description'], title='Scatter Plot: Star Rating vs. Number of Reviews')

        fig.update_layout(xaxis_title='Star Rating', yaxis_title='Number of Reviews')
        fig.show()

    def convert_to_amazon_url(self, product_url):
        # Parse the query parameters from the URL
        parsed_url = urlparse(product_url)
        query_params = parse_qs(parsed_url.query)

        # Extract the 'url' parameter, which contains the actual Amazon URL
        amazon_url = query_params.get('url', [None])[0]

        if amazon_url:
            # Append additional parameters or modify the URL as needed
            # For example, you might want to add '/ref=sr_1_2?' to simulate the structure of Amazon URLs
            modified_url = amazon_url + '/ref=sr_1_2?'

            # Reconstruct the URL with the modified parameters
            reconstructed_url = urlunparse(parsed_url._replace(path='', query=f'url={modified_url}', fragment=''))

            # Ensure the URL starts with 'https://www.amazon.it/'
            reconstructed_url = urljoin('https://www.amazon.it/', reconstructed_url)

            return reconstructed_url

        return None

    def preprocess_descriptions(self, filter_descriptions=False):
        """
        Preprocesses product descriptions from self.data (if scraped) or an existing TSV file.

        :param filter_descriptions: (bool) If True, removes duplicate descriptions based on their text.
        :return: list of unique processed descriptions (tokenized lists).
        """
        processed_descriptions = []
        unique_descriptions = set()  # Track unique tokenized descriptions

        # Check if data has been loaded into self.df
        if self.df is not None:
            # Process descriptions from the DataFrame loaded from file
            for description in self.df['Description']:
                processed_tokens = self.text_preprocessor.preprocess_text(description)
                description_tuple = tuple(processed_tokens)  # Convert to tuple for hashability

                if not filter_descriptions or description_tuple not in unique_descriptions:
                    processed_descriptions.append(processed_tokens)
                    unique_descriptions.add(description_tuple)
        elif self.data:
            # Process descriptions from self.data (directly scraped)
            for product in self.data:
                description = product[0]  # Description is the first item in each data entry
                processed_tokens = self.text_preprocessor.preprocess_text(description)
                description_tuple = tuple(processed_tokens)  # Convert to tuple for hashability

                if not filter_descriptions or description_tuple not in unique_descriptions:
                    processed_descriptions.append(processed_tokens)
                    unique_descriptions.add(description_tuple)
        else:
            print("No data to process. Please scrape or load data first.")

        print(f"Processed {len(processed_descriptions)} unique descriptions.")
        return processed_descriptions

