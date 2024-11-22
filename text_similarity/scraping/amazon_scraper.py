import os
import time
from datetime import datetime
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from text_similarity.scraping.data_processing import load_dataset, save_to_tsv
from text_similarity.scraping.visualization import plot_top_products, scatter_plot
from utils.utils import get_headers

class AmazonScraper:
    def __init__(self, keywords, num_pages):
        self.scraped_results = None
        self.keywords = keywords
        self.num_pages = num_pages
        self.base_url = "https://www.amazon.it/s"
        self.data = []

    def scrape_amazon_products(self):
        """
        Scrapes Amazon search results for specified keywords and number of pages.
        """
        unique_entries = set()  # Track unique products to avoid duplicates
        total_products = 0
        new_products = 0

        for keyword in self.keywords.split(","):
            keyword = keyword.strip()
            print(f"Scraping keyword '{keyword}'...")
            for page in range(1, self.num_pages + 1):
                url = f"{self.base_url}?k={keyword}&page={page}"
                headers = get_headers()
                response = requests.get(url, headers=headers)
                time.sleep(2)  # Avoid being blocked

                if response.status_code == 200:
                    print(f"Scraping page {page}...")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    products = soup.find_all('div', {'data-component-type': 's-search-result'})

                    for product in products:
                        try:
                            product_data = self.extract_product_data(product)
                            if not product_data:
                                continue

                            # Deduplicate products based on unique attributes
                            product_tuple = tuple(product_data.values())
                            if product_tuple in unique_entries:
                                continue
                            unique_entries.add(product_tuple)
                            self.data.append(product_data)
                            new_products += 1
                            total_products += 1
                        except Exception as e:
                            print(f"Error processing product: {e}")

                else:
                    print(f"Failed to retrieve page {page} for keyword '{keyword}'. Status: {response.status_code}")
                    break

            print(f"Completed scraping for keyword '{keyword}'.")
        print(f"Scraping completed. Total: {total_products}, New: {new_products}")

    def extract_product_data(self, product):
        """
        Extracts product details from a BeautifulSoup product element.
        """
        try:
            # Extract product description
            description_element = product.find('span', class_='a-size-base-plus a-color-base a-text-normal')
            product_description = description_element.text.strip() if description_element else None

            # Extract product URL
            url_element = product.find('a', class_='a-link-normal s-no-outline')
            product_url = urljoin(self.base_url, url_element['href']) if url_element else None

            # Extract price
            price_container = product.find('span', class_='a-price')
            price_element = price_container.find('span', class_='a-offscreen') if price_container else None
            product_price = (
                float(price_element.text.strip().replace('.', '').replace(',', '.').replace('€', ''))
                if price_element else None
            )

            # Extract star rating
            star_element = product.find('i', class_='a-icon-star-small')
            star_rating = (
                float(star_element.find('span', class_='a-icon-alt').text.split()[0].replace(',', '.'))
                if star_element else None
            )

            # Extract review count
            review_count_element = product.find('span', class_='a-size-base s-underline-text')
            num_reviews = (
                int(review_count_element.text.strip().replace('.', '').replace(',', ''))
                if review_count_element else None
            )

            return {
                'description': product_description,
                'price': product_price,
                'url': product_url,
                'star_rating': star_rating,
                'reviews': num_reviews,
            }
        except Exception as e:
            print(f"Error extracting product data: {e}")
            return None

    def save_results(self, output_dir="data/raw"):
        """
        Saves the scraped data to a TSV file.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{self.keywords.split(',')[0]}_results_{datetime.now().strftime('%Y-%m-%d')}.tsv"
        file_path = os.path.join(output_dir, file_name)
        save_to_tsv(self.data, file_path)
        print(f"Data saved to {file_path}")

    def load_results(self, file_path):
        """
        Loads data from a TSV file into a DataFrame.
        """
        return load_dataset(file_path)

    def analyze_data(self, df):
        """
        Analyzes the scraped data and visualizes insights.
        """
        # Top 10 products by star rating
        top_10_ratings = df.nlargest(10, 'star_rating')
        plot_top_products(top_10_ratings, 'star_rating', 'Top 10 Products by Rating')

        # Scatter plot of price vs. star rating
        scatter_plot(df, x='price', y='star_rating', title='Price vs. Star Rating')
