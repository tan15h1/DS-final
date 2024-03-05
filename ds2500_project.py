#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 18:34:40 2023

Tanishi Datta & Gianni Diarbi
DS2500
Fall 2023
Final Project Code

@author: tanishidatta
"""

import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to load and clean the dataset
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    return data

# Function to filter data based on date, drop NaN values, 
# and filter short narratives
def filter_data(data):
    data = data[data['Date received'] >= '2020-01-01']
    data = data.dropna(subset=['Consumer complaint narrative'])
    data = data[data['Consumer complaint narrative'].apply(lambda x: \
                                                    len(str(x).split()) > 3)]
    return data

# Function to perform sentiment analysis using TextBlob
def perform_sentiment_analysis(data):
    data['Sentiment'] = data['Consumer complaint narrative'].apply(lambda x: \
                                        TextBlob(str(x)).sentiment.polarity)
    return data

# Function to create or overwrite a CSV file with top 10 worst bank reviews
def create_worst_bank_reviews_csv(data, \
    filename = 'Top_10_Worst_Bank_Reviews.csv', num_top_banks = 10):
    avg_sentiment_by_bank = data.groupby('Company')['Sentiment'].mean()
    sorted_banks = avg_sentiment_by_bank.sort_values(ascending=True)
    top_worst_banks = sorted_banks.head(num_top_banks)
    
    # Create a DataFrame with bank names, comments, and 
    # consumer trust scores for the worst banks
    worst_banks_df = pd.DataFrame({
        'Bank Name': top_worst_banks.index,
        'Comment': data.groupby('Company')\
            ['Consumer complaint narrative'].apply(lambda x: 
                            '\n'.join(x)).loc[top_worst_banks.index].values,
                                'Consumer Trust Score': top_worst_banks.values
    })

    # Save the DataFrame to a CSV file, replacing previous file if it exists
    worst_banks_df.to_csv(filename, mode = 'w', index = False)

    print(f"CSV file '{filename}' created and saved in the current \
          working directory.")

# Function to perform regression analysis & plot the top/bottom trusted banks
def perform_regression_analysis(data, num_top_banks = 10, \
                                num_bottom_banks = 10):
    avg_sentiment_by_bank = data.groupby('Company')['Sentiment'].mean()
    sorted_banks = avg_sentiment_by_bank.sort_values(ascending = False)

    # Extract data for regression analysis
    top_banks = sorted_banks.head(num_top_banks)
    bottom_banks = sorted_banks.tail(num_bottom_banks)

    # Combine top and bottom banks
    selected_banks = pd.concat([top_banks, bottom_banks])

    # Fit a linear regression model
    regression_model = np.polyfit(range(len(selected_banks)), \
                                  selected_banks.values, 1)

    # Create regression line values
    regression_line = np.polyval(regression_model, range(len(selected_banks)))

    # Plot the data and regression line
    plt.figure(figsize = (12, 6))
    sns.barplot(x=selected_banks.index, y = selected_banks.values, \
                color = 'lightblue')
    plt.plot(selected_banks.index, regression_line, color = 'red', \
             linestyle = '--', linewidth = 2, label = 'Regression Line')
    plt.xticks(rotation = 45, ha = 'right')
    plt.xlabel('Bank Name')
    plt.ylabel('Consumer Trust Score')
    plt.title("Consumer Trust Scores of Top 10 and Bottom 10 Banks")
    plt.tight_layout()
    plt.show()

def main(file_path = 'bank_complaints_real.csv'):
    # Load and clean the data
    data = load_and_clean_data(file_path)

    # Filter data based on date, drop NaN values, and filter short narratives
    filtered_data = filter_data(data)

    # Perform sentiment analysis
    filtered_data = perform_sentiment_analysis(filtered_data)

    # Print the average sentiment score for each bank
    avg_sentiment_by_bank = \
        filtered_data.groupby('Company')['Sentiment'].mean()
    print("Average Sentiment by Bank:\n", avg_sentiment_by_bank)

    # Create or overwrite the CSV file of top 10 worst bank reviews
    create_worst_bank_reviews_csv(filtered_data)

    # Perform regression analysis and plot the top and bottom trusted banks
    perform_regression_analysis(filtered_data)

    # Identify the most trusted bank based on average sentiment score
    most_trusted_bank = avg_sentiment_by_bank.idxmax()
    print("\nMost Trusted Bank:", most_trusted_bank)

if __name__ == "__main__":
    main()