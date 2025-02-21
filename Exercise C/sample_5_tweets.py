import pandas as pd

def sample_tweets(file_path, sample_size=20):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Ensure sentiment column exists
    if 'sentiment' not in df.columns:
        raise ValueError("The dataset must contain a 'sentiment' column.")
    
    # Define the number of samples per sentiment class
    num_samples_per_class = sample_size // 3
    remainder = sample_size % 3
    
    # Sample approximately equal number of rows from each sentiment category
    sampled_df = (df.groupby('sentiment', group_keys=False)
                    .apply(lambda x: x.sample(n=min(len(x), num_samples_per_class), random_state=42)))
    
    # If there is a remainder, sample additional rows from random sentiment groups
    if remainder > 0:
        additional_samples = df.sample(n=remainder, random_state=42)
        sampled_df = pd.concat([sampled_df, additional_samples])
    
    # Shuffle the final sampled dataset
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return sampled_df

# Usage example
file_path = "example_tweets.csv"  # Adjust path if needed
sampled_5_tweets = sample_tweets(file_path, 5)

# Save to a new CSV file
sampled_5_tweets.to_csv("sampled_5_tweets.csv", index=False)