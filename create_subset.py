import pandas as pd

# CONFIGURATION
# 1. Download the dataset from: https://www.kaggle.com/datasets/kazanova/sentiment140
# 2. Extract the ZIP file.
# 3. Place 'training.1600000.processed.noemoticon.csv' in the project folder.
INPUT_FILE = 'training.1600000.processed.noemoticon.csv'
OUTPUT_FILE = 'data/rome_weather_sentiment140_subset.csv'
KEYWORDS = ['Rome', 'Roma', 'Italy', 'Italia'] # Filter for these words
SAMPLE_SIZE = 5000  # Number of rows to save if filtering returns too many

def create_subset():
    print(f"Reading {INPUT_FILE}...")
    try:
        # Sentiment140 has no headers, so we define them
        # 0: target, 1: id, 2: date, 3: flag, 4: user, 5: text
        cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
        
        # Read the large file (encoding is often latin-1 for this dataset)
        df = pd.read_csv(INPUT_FILE, encoding='latin-1', names=cols)
        print(f"Full dataset size: {len(df)} rows")

        # Filter for keywords with word boundaries to avoid matching "Chrome" or "Vitality"
        # We use regex=True and \b boundary markers
        pattern = r'\b(' + '|'.join(KEYWORDS) + r')\b'
        print(f"Filtering for pattern: {pattern}...")
        
        df_subset = df[df['text'].str.contains(pattern, case=False, na=False, regex=True)]
        print(f"Found {len(df_subset)} relevant tweets.")

        # If we found relevant tweets, take a sample if it's huge
        if len(df_subset) > SAMPLE_SIZE:
             df_subset = df_subset.sample(n=SAMPLE_SIZE, random_state=42)
             print(f"Taking a random sample of {SAMPLE_SIZE} tweets.")
        elif len(df_subset) == 0:
            print("No tweets found with those keywords! Saving a random generic sample instead.")
            df_subset = df.sample(n=SAMPLE_SIZE, random_state=42)

        # Save to CSV
        df_subset.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! Saved subset to {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'.")
        print("Please download it from Kaggle and place it in this folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_subset()
