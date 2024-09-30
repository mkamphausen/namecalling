import pandas as pd
import re
from collections import Counter

# Function to extract biden-related hashtags
def extract_biden_hashtags(text):
    return re.findall(r'#\w*biden\w*', text.lower())

# Function to extract trump-related hashtags
def extract_trump_hashtags(text):
    return re.findall(r'#\w*trump\w*', text.lower())

# General function to analyze hashtags and their occurrence in name-calling tweets
def analyze_hashtags(df, extract_function):
    # Initialize counters for hashtags
    hashtag_counter = Counter()
    name_calling_counter = Counter()
    not_name_calling_counter = Counter()

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        hashtags = extract_function(row['tweet'])  # Use the passed extract function
        is_name_calling = row['name_calling_pred']  # 1 or 0

        # Update counters
        for hashtag in hashtags:
            hashtag_counter[hashtag] += 1
            if is_name_calling == 1:
                name_calling_counter[hashtag] += 1
            else:
                not_name_calling_counter[hashtag] += 1

    # Prepare the results as a DataFrame
    result = pd.DataFrame({
        'hashtag': list(hashtag_counter.keys()),
        'total_occurrences': list(hashtag_counter.values()),
        'name_calling': [name_calling_counter[h] for h in hashtag_counter],
        'not_name_calling': [not_name_calling_counter[h] for h in hashtag_counter]
    })

    return result

# Function to filter out hashtags with more than one occurrence and sort by total occurrences
def filter_and_sort_hashtags(df):
    filtered_df = df

    # Sort by total occurrences in descending order
    sorted_df = filtered_df.sort_values(by='total_occurrences', ascending=False)

    return sorted_df

# Load the CSV file into a DataFrame
df = pd.read_csv('data/all_tweets_without_duplicates.csv', sep=',', encoding='utf-8')

# Analyze Biden-related hashtags
biden_hashtag_analysis = analyze_hashtags(df, extract_biden_hashtags)

# Analyze Trump-related hashtags
trump_hashtag_analysis = analyze_hashtags(df, extract_trump_hashtags)

# Filter and sort both analyses
filtered_sorted_biden = filter_and_sort_hashtags(biden_hashtag_analysis)
filtered_sorted_trump = filter_and_sort_hashtags(trump_hashtag_analysis)

# Save both analyses to separate CSV files
filtered_sorted_biden.to_csv('data/biden_processed/biden_hashtags.csv', index=False)
filtered_sorted_trump.to_csv('data/trump_processed/trump_hashtags.csv', index=False)

print(f"Analysis completed. {len(filtered_sorted_biden)} Biden-related hashtags and {len(filtered_sorted_trump)} Trump-related hashtags with more than one occurrence.")