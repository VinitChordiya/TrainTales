import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(sentiment_df):
    # """
    # Analyzes the sentiment of each sentence in the given DataFrame using the SentimentIntensityAnalyzer.

    # Args:
    #     sentiment_df (pandas.DataFrame): The DataFrame containing the sentences to analyze.

    # Returns:
    #     pandas.DataFrame: The original DataFrame with additional columns for sentiment scores.

    # """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_df['sentence'].apply(sia.polarity_scores)
    sentiment_scores = sentiment_scores
    sentiment_df = pd.concat([sentiment_df, sentiment_scores.apply(pd.Series)], axis=1)

    # # Initialize sentiment analyzer
    # sia = SentimentIntensityAnalyzer()

    # # Analyze sentiment for each sentence
    # sentiment_data = []
    # for sentence in sen_count:
    #     sentiment_scores = sia.polarity_scores(sentence)
    #     sentiment_data.append({
    #         'sentence': sentence,
    #         'negative': sentiment_scores['neg'],
    #         'neutral': sentiment_scores['neu'],
    #         'positive': sentiment_scores['pos'],
    #         'compound': sentiment_scores['compound']
    #     })
    return sentiment_df

def categorize_sentiment(compound_score):
    # """
    # Categorizes the sentiment based on the compound score.

    # Parameters:
    # compound_score (float): The compound score representing the sentiment.

    # Returns:
    # str: The sentiment category ('positive', 'negative', or 'neutral').
    # """
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'
