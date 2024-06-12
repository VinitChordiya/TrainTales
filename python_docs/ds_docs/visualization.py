import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image

def plot_category_distribution(sentiment_df):
    # """
    # Plots the distribution of sentiment categories in a given DataFrame.

    # Parameters:
    # sentiment_df (DataFrame): The DataFrame containing the sentiment data.

    # Returns:
    # None
    # """
    sns.countplot(x='category', data=sentiment_df)
    plt.title('Sentiment Category Distribution')
    plt.show()
    
def plot_score_distribution(sentiment_df):
    # """
    # Plots the distribution of compound scores from a sentiment dataframe.

    # Parameters:
    # sentiment_df (pandas.DataFrame): The dataframe containing the sentiment scores.

    # Returns:
    # None
    # """
    sns.histplot(sentiment_df['compound'], bins=30, kde=True)
    plt.title('Distribution of Compound Scores')
    plt.show()
    
def plot_daily_sentiment(sentiment_df):
    # """
    # Plots the daily average sentiment score over time.
    
    # Parameters:
    # sentiment_df (DataFrame): The DataFrame containing sentiment scores.
    
    # Returns:
    # None
    # """
    # Extract the numeric columns from the DataFrame
    numeric_df = sentiment_df[['neg', 'neu', 'pos', 'compound']]
    
    # Resample the DataFrame to get the daily average sentiment scores
    daily_sentiment = numeric_df.resample('D').mean()
    
    # Create a figure with a size of 12x6 inches
    plt.figure(figsize=(12, 6))
    
    # Plot the daily average compound score
    daily_sentiment['compound'].plot()
    
    plt.title('Daily Average Sentiment Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Compound Score')
    plt.grid(True)
    plt.show()

def plot_ind_wordcloud(complaints):
    """
    Generates and displays a word cloud based on the given complaints.
    
    Parameters:
    complaints (str): The text containing the complaints.

    Returns:
    None
    """

    image_path = "ind3.png"
    # Load the image mask
    # # Debugging prints
    # print(f"Image path: {image_path}")
    # try:
    #     # Load the image mask
    #     mask = np.array(Image.open(image_path))
    #     print("Image loaded successfully")
    # except Exception as e:
    #     print(f"Error loading image: {e}")
    #     return
    mask = np.array(Image.open(image_path))

    # Create a word cloud object
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=200,
        mask=mask,
        stopwords=set(WordCloud().stopwords)
    ).generate(complaints)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
