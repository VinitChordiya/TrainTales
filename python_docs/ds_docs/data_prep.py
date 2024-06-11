import pandas as pd
from datetime import datetime, timedelta
from nltk.tokenize import sent_tokenize


def load_data(file_path):
    # """
    # Loads data from a file.

    # Args:
    #     file_path (str): The path to the file.

    # Returns:
    #     str: The content of the file.

    # Raises:
    #     FileNotFoundError: If the file does not exist.
    #     IOError: If there is an error reading the file.
    # """
    with open(file_path, "r") as file:
        return file.read()
def preprocess_data(complaints):
    # """
    # Preprocesses the given complaints by tokenizing them into sentences and creating a DataFrame with sentiment data.

    # Args:
    #     complaints (str): The complaints to be preprocessed.

    # Returns:
    #     pandas.DataFrame: A DataFrame containing sentiment data for each sentence in the complaints.
    # """
    sentences = sent_tokenize(complaints)
    
    sentiment_data = [{'sentence': sentence} for sentence in sentences]
    return pd.DataFrame(sentiment_data)
def add_dummy_timestamps(sentiment_df):
    # """
    # Adds dummy timestamps to a DataFrame.

    # Parameters:
    # sentiment_df (pandas.DataFrame): The DataFrame to add timestamps to.

    # Returns:
    # pandas.DataFrame: The DataFrame with dummy timestamps added.
    # """
    start_date = datetime.now()
    timestamps = [start_date - timedelta(days=i) for i in range(len(sentiment_df))]
    sentiment_df['timestamp'] = timestamps
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
    sentiment_df.set_index('timestamp', inplace=True)
    return sentiment_df
