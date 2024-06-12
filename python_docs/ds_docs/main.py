from data_prep import load_data, add_dummy_timestamps, preprocess_data
from sentiment_analysis import analyze_sentiment, categorize_sentiment
from visualization import plot_category_distribution, plot_score_distribution, plot_daily_sentiment, plot_ind_wordcloud
import numpy as np
#////////////////////


def main():
    # 
    # The main function of the program.

    # This function loads and preprocesses data from a file, analyzes the sentiment of the data,
    # categorizes the sentiment, adds dummy timestamps, and plots and visualizes the data.

    # Returns:
    #     None
    
    # Load and preprocess data
    file_path = "data1.txt"
    complaints = load_data(file_path)
    complaints_wc = load_data(file_path)

    sentiment_df = preprocess_data(complaints)
    
    # Analyze sentiment
    sentiment_df = analyze_sentiment(sentiment_df)
    
    # Categorize sentiment
    sentiment_df['category'] = sentiment_df['compound'].apply(categorize_sentiment)
    print(sentiment_df.to_string())
    # Add dummy timestamps
    sentiment_df = add_dummy_timestamps(sentiment_df)
    
    # Plot and visualize data
    plot_ind_wordcloud(complaints_wc)
    plot_category_distribution(sentiment_df)
    plot_score_distribution(sentiment_df)
    plot_daily_sentiment(sentiment_df)
if __name__ == "__main__":
    main()
