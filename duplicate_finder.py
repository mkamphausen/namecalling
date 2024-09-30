import pandas as pd

def clean_and_merge(file1, file2):
    # Read the two CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Count and remove rows with NaN in 'tweet' column for both files
    nan_count_df1 = df1['tweet'].isna().sum()
    nan_count_df2 = df2['tweet'].isna().sum()

    df1_clean = df1.dropna(subset=['tweet'])
    df2_clean = df2.dropna(subset=['tweet'])

    total_nan_count = nan_count_df1 + nan_count_df2

    # Merge the two DataFrames on the 'tweet' column
    merged_df = pd.concat([df1_clean, df2_clean])

    # Count duplicates before removing them
    duplicates_count = merged_df.duplicated(subset=['tweet_id']).sum()

    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['tweet_id'], keep='first')

    # Print statistics
    print(f"Number of NaN rows deleted: {total_nan_count}")
    print(f"Number of duplicate tweets deleted: {duplicates_count}")
    print(f"Number of unique tweets after merging: {merged_df.shape[0]}")

    return merged_df

def save_to_csv(df, output_file):
    """Save the DataFrame to a CSV file."""
    df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")


# Example usage
file1 = 'data/biden_processed/biden_with_predictions.csv' # Replace with the path to your first CSV file
file2 = 'data/trump_processed/trump_with_predictions.csv' # Replace with the path to your second CSV file
output_file = 'data/all_tweets_without_duplicates.csv'         # Specify the path for the output file

merged_df = clean_and_merge(file1, file2)

save_to_csv(merged_df, output_file)
