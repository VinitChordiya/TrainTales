import seaborn as sns
import matplotlib.pyplot as plt

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
