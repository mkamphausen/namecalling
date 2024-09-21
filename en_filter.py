import pandas as pd
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from concurrent.futures import ThreadPoolExecutor

# Function to clean the tweet by temporarily removing hashtags and URLs (without modifying the original tweet)
def clean_for_detection(text):
    # Temporarily remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Temporarily remove hashtags
    text = re.sub(r'#\w+', '', text)
    return text.strip()

# Function to detect language and filter only English tweets
def is_english(text):
    try:
        cleaned_text = clean_for_detection(text)  # Clean the tweet temporarily for language detection
        return detect(cleaned_text) == 'en'
    except LangDetectException:
        return False

# Load the CSV file into a DataFrame
df = pd.read_csv('data/US Election 2020 Tweets/hashtag_donaldtrump.csv', sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')

# Specify the column containing the tweet text
tweet_column = 'tweet'  

# Drop rows with NaN values in the specified column
df = df.dropna(subset=[tweet_column])

# Apply the filtering function to the DataFrame using ThreadPoolExecutor for faster processing
with ThreadPoolExecutor(max_workers=16) as executor:
    df['is_english'] = df[tweet_column].apply(is_english)

# Filter the DataFrame to keep only English tweets
df_filtered = df[df['is_english']]

# Drop the temporary column used for filtering
df_filtered = df_filtered.drop(columns=['is_english'])

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('data/trump_processed/trump_filtered_english.csv', index=False)

print(f"Filtered {len(df_filtered)} English tweets from the original dataset.")