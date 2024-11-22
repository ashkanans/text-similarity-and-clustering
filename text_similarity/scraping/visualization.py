import plotly.express as px
from matplotlib import pyplot as plt


def plot_top_products(df, column, title):
    """
    Plots a horizontal bar chart of the top 10 products based on a specified column.

    :param df: pandas.DataFrame - DataFrame containing the data.
    :param column: str - Column name to sort and display (e.g., 'star_rating', 'price').
    :param title: str - Title of the chart.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    top_products = df.nlargest(10, column)
    fig = px.bar(
        top_products,
        x=column,
        y='description',
        orientation='h',
        title=title,
        text=column
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis=dict(type='category'), xaxis_title=column.capitalize(), yaxis_title='Product Description')
    fig.show()


def scatter_plot(df, x, y, title):
    """
    Plots a scatter plot with size and color encoding.

    :param df: pandas.DataFrame - DataFrame containing the data.
    :param x: str - Column name for the x-axis.
    :param y: str - Column name for the y-axis.
    :param title: str - Title of the scatter plot.
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' or '{y}' not found in DataFrame.")

    fig = px.scatter(
        df,
        x=x,
        y=y,
        size='reviews',  # Bubble size based on the number of reviews
        color='star_rating',  # Color based on star rating
        hover_data=['description'],
        title=title
    )
    fig.update_layout(xaxis_title=x.capitalize(), yaxis_title=y.capitalize())
    fig.show()


def price_distribution_boxplot(df, price_column, category_column):
    """
    Creates a box plot to visualize price distribution across categories.

    :param df: pandas.DataFrame - DataFrame containing the data.
    :param price_column: str - Column name for prices.
    :param category_column: str - Column name for categories (e.g., price ranges).
    """
    if price_column not in df.columns or category_column not in df.columns:
        raise ValueError(f"Columns '{price_column}' or '{category_column}' not found in DataFrame.")

    fig = px.box(
        df,
        x=category_column,
        y=price_column,
        points="all",
        title="Price Distribution Across Categories",
    )
    fig.update_layout(xaxis_title=category_column.capitalize(), yaxis_title=price_column.capitalize())
    fig.show()


def plot_token_length_distribution(df):
    """
    Plots the distribution of token lengths.
    """
    token_lengths = df['Description'].apply(lambda x: len(x.split())).tolist()

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=range(1, max(token_lengths) + 2), edgecolor='black')
    plt.title("Distribution of Token Lengths in Descriptions")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.xticks(range(1, max(token_lengths) + 2))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
