import pandas as pd
import re
from collections import Counter

def extract_hashtags(text):
    """Extract hashtags from a text string."""
    return re.findall(r'#\w+', str(text))

def analyze_hashtags(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Check if the required columns exist
    if 'tweet' not in df.columns or 'name_calling_pred' not in df.columns:
        print("Error: The CSV file must contain 'tweet' and 'name_calling_pred' columns.")
        return

    # Initialize counters for hashtags in all tweets, and specific occurrences for name calling and not name calling
    hashtag_counter = Counter()
    pred_hashtag_counts = {'0': Counter(), '1': Counter()}

    # Iterate through each row to extract hashtags and count occurrences based on `name_calling_pred`
    for _, row in df.iterrows():
        hashtags = extract_hashtags(row['tweet'])
        hashtag_counter.update(hashtags)

        # Check if `name_calling_pred` column has 0 or 1 and update respective counters
        if row['name_calling_pred'] == 0:
            pred_hashtag_counts['0'].update(hashtags)
        elif row['name_calling_pred'] == 1:
            pred_hashtag_counts['1'].update(hashtags)

    # Create a DataFrame to display the results in table format
    result = pd.DataFrame({
        'hashtag': list(hashtag_counter.keys()),
        'overall_occurrence': list(hashtag_counter.values()),
        'name_calling': [pred_hashtag_counts['1'].get(tag, 0) for tag in hashtag_counter.keys()],
        'not_name_calling': [pred_hashtag_counts['0'].get(tag, 0) for tag in hashtag_counter.keys()]
    })

    return result

    # Display the results in a table format
def display_results():
    print("\nHashtag Analysis Table:")
    print(result.to_string(index=False))

def filter_and_sort_hashtags(result_df):
    """Sort the hashtags by overall occurrence and filter those with more than one occurrence."""
    # Filter hashtags that occur more than once
    filtered_df = result_df[result_df['overall_occurrence'] > 1]
    
    # Sort by overall occurrence in descending order
    sorted_df = filtered_df.sort_values(by='overall_occurrence', ascending=False)
    
    # Display the sorted DataFrame
    print("\nHashtags with more than one occurrence (sorted by overall occurrence):")
    print(sorted_df.to_string(index=False))

# Example usage
file_path = 'trump_output_with_predictions.csv'  # Replace with the path to your actual CSV file
result_df = analyze_hashtags(file_path)

filter_and_sort_hashtags(result_df)
