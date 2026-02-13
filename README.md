
# Weather Sentiment Analysis Project

A beginner-friendly NLP project that classifies how people feel about the weather (positive, negative, or neutral) using machine learning and pre-trained models.

## 📋 Overview

This project demonstrates core NLP concepts including:
- Text preprocessing and cleaning
- Feature extraction (TF-IDF)
- Sentiment classification using multiple approaches
- Model evaluation and comparison
- Data visualization

## 🎯 Features

- **Multiple Models**: Naive Bayes, Logistic Regression, and VADER Sentiment Analyzer
- **Text Preprocessing**: Tokenization, lowercasing, stopword removal, lemmatization
- **Data Visualization**: Confusion matrices, accuracy comparisons, sample predictions
- **Easy to Use**: Simple API for classifying new text
- **Well Documented**: Clear code with examples

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/sentiment-analysis-project.git
cd sentiment-analysis-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python main.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook sentiment_analysis_demo.ipynb
```

## 📂 Project Structure

```
sentiment-analysis-project/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── main.py                        # Main analysis script
├── sentiment_analyzer.py          # Core sentiment analysis module
├── data/
│   ├── sample_reviews.csv         # Sample movie reviews dataset
│   └── predictions/               # Output predictions
├── notebooks/
│   └── sentiment_analysis_demo.ipynb  # Interactive demo
├── results/
│   └── evaluation_results.txt     # Model evaluation metrics
└── .gitignore                     # Git ignore file
```


## 📊 Dataset

The project uses a sample dataset of weather-related comments and their sentiment. You can:
- Use the provided `weather_sentiment_samples.csv`
- Replace with your own dataset (CSV format with 'text' and 'label' columns)

### Expected Format
```csv
text,label
"It's a beautiful sunny day!",positive
"Rainy days make me feel gloomy.",negative
"It's okay outside, I guess.",neutral
```

## 🧠 Models Used

### 1. Naive Bayes
- Fast training and prediction
- Good baseline for text classification
- Works well with TF-IDF features

### 2. Logistic Regression
- Interpretable predictions
- Handles high-dimensional data well
- Provides probability estimates

### 3. VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Pre-trained lexicon-based approach
- No training required
- Great for social media and casual text

## 📈 Results

Example output from model comparison:
```
Model Evaluation Results:
========================
Naive Bayes Accuracy:       87.5%
Logistic Regression Accuracy: 89.2%
VADER Accuracy:             82.1%

Best Model: Logistic Regression
```

## 💻 Usage Example

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Classify text
text = "This product is amazing! Love it!"
sentiment = analyzer.predict(text)
print(f"Sentiment: {sentiment}")  # Output: positive

# Get confidence scores
scores = analyzer.predict_with_scores(text)
print(scores)
```

## 🔍 How It Works

1. **Data Loading**: Load reviews from CSV file
2. **Preprocessing**: Clean and normalize text
3. **Feature Extraction**: Convert text to TF-IDF vectors
4. **Training**: Train ML models on labeled data
5. **Evaluation**: Test on validation set
6. **Comparison**: Compare model performance
7. **Prediction**: Classify new text

## 🛠️ Technologies

- **Python 3.8+**
- **scikit-learn**: Machine learning models
- **NLTK**: Natural Language Toolkit
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization

## 📚 Learning Resources

- [NLTK Book](https://www.nltk.org/book/)
- [scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [NLP with Python](https://realpython.com/sentiment-analysis-python/)

## 🎓 Portfolio Value

This project demonstrates:
- ✅ Understanding of NLP fundamentals
- ✅ Practical ML model implementation
- ✅ Data preprocessing and cleaning
- ✅ Model evaluation and comparison
- ✅ Code organization and documentation
- ✅ Git version control
- ✅ Data visualization

## 📝 Future Improvements

- [ ] Add deep learning models (LSTM, BERT)
- [ ] Implement cross-validation
- [ ] Add more advanced preprocessing
- [ ] Create REST API endpoint
- [ ] Add real-time sentiment tracking
- [ ] Expand to multi-class sentiment (5-star ratings)

## 📄 License

MIT License - feel free to use this project for your portfolio

## 👤 Author

Your Name - [Your GitHub Profile](https://github.com/your-username)

---

**Made with ❤️ for data science portfolio**
