# Rome Sentiment Analysis from Twitter data 

A data analysis project exploring sentiment in social media (Twitter) regarding **Rome** during Spring 2009.

## Overview
This project performs sentiment analysis on a subset of the **Sentiment140** dataset, specifically filtering for tweets related to "Rome". 

During the analysis, I discovered that the dataset captures a significant historical event: the **L'Aquila Earthquake** (April 6, 2009), which heavily influenced public sentiment.

##  Key Features
- **Data Filtering**: Extracted ~1,000 Rome-specific tweets from the 1.6 million tweet Sentiment140 dataset.
- **Sentiment Classification**: Categorized tweets as Positive, Negative, or Neutral.
- **Word Clouds**: Visualized common terms in positive vs. negative tweets (highlighting "sad", "earthquake" vs. "love", "great").
- **Time Series Analysis**: Identified a massive spike in negative sentiment corresponding exactly to the date of the 2009 earthquake.


##  Key Findings
- **Positive Sentiment**: "Great", "Love", "Day" were common.
- **Negative Sentiment**: "Miss", "Sad", "Earthquake" surged on April 6, 2009.
- **Event Detection**: The sentiment timeline perfectly maps to real-world events.

##  How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the notebook:
   ```bash
   jupyter notebook rome_sentiment_analysis.ipynb
   ```
