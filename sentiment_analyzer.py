"""
Sentiment Analysis Module

This module provides functionality to classify text sentiment using multiple approaches:
1. Naive Bayes with TF-IDF
2. Logistic Regression with TF-IDF
3. VADER Sentiment Analyzer
"""

import string
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    A comprehensive sentiment analyzer using multiple machine learning approaches.
    
    Attributes:
        preprocessor: Text preprocessing pipeline
        vectorizer: TF-IDF vectorizer for feature extraction
        nb_model: Naive Bayes classifier
        lr_model: Logistic Regression classifier
        vader_analyzer: VADER sentiment analyzer
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer with all models."""
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.nb_model = None
        self.lr_model = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.is_trained = False
        
    def preprocess(self, text):
        """
        Preprocess text for analysis.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned and normalized text
        """
        return self.preprocessor.preprocess(text)
    
    def train(self, texts, labels):
        """
        Train the sentiment models on labeled data.
        
        Args:
            texts (list): List of text samples
            labels (list): List of corresponding sentiment labels (positive/negative/neutral)
            
        Returns:
            dict: Training performance metrics
        """
        # Preprocess texts
        processed_texts = [self.preprocess(text) for text in texts]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Convert labels to numeric
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        y = np.array([label_map.get(label.lower(), 1) for label in labels])
        
        # Train Naive Bayes
        self.nb_model = MultinomialNB()
        self.nb_model.fit(X, y)
        
        # Train Logistic Regression
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_model.fit(X, y)
        
        self.is_trained = True
        
        return {
            'status': 'success',
            'samples_trained': len(texts),
            'nb_accuracy': self.nb_model.score(X, y),
            'lr_accuracy': self.lr_model.score(X, y)
        }
    
    def predict(self, text, model='logistic'):
        """
        Predict sentiment of given text.
        
        Args:
            text (str): Text to classify
            model (str): Which model to use ('naive_bayes', 'logistic', or 'vader')
            
        Returns:
            str: Predicted sentiment (positive/negative/neutral)
        """
        if model == 'vader':
            return self._vader_predict(text)
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        processed_text = self.preprocess(text)
        X = self.vectorizer.transform([processed_text])
        
        if model == 'logistic':
            pred = self.lr_model.predict(X)[0]
        elif model == 'naive_bayes':
            pred = self.nb_model.predict(X)[0]
        else:
            raise ValueError("Model must be 'logistic', 'naive_bayes', or 'vader'")
        
        # Convert numeric label back to string
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return label_map[pred]
    
    def predict_with_scores(self, text, model='logistic'):
        """
        Predict sentiment with confidence scores.
        
        Args:
            text (str): Text to classify
            model (str): Which model to use
            
        Returns:
            dict: Predicted sentiment and confidence scores
        """
        if model == 'vader':
            return self._vader_predict_with_scores(text)
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        processed_text = self.preprocess(text)
        X = self.vectorizer.transform([processed_text])
        
        if model == 'logistic':
            probas = self.lr_model.predict_proba(X)[0]
            pred = self.lr_model.predict(X)[0]
        elif model == 'naive_bayes':
            probas = self.nb_model.predict_proba(X)[0]
            pred = self.nb_model.predict(X)[0]
        else:
            raise ValueError("Model must be 'logistic', 'naive_bayes', or 'vader'")
        
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        return {
            'sentiment': label_map[pred],
            'scores': {
                'negative': float(probas[0]),
                'neutral': float(probas[1]),
                'positive': float(probas[2])
            }
        }
    
    def _vader_predict(self, text):
        """Predict sentiment using VADER analyzer."""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _vader_predict_with_scores(self, text):
        """Predict sentiment with VADER scores."""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'scores': {
                'negative': float(scores['neg']),
                'neutral': float(scores['neu']),
                'positive': float(scores['pos']),
                'compound': float(scores['compound'])
            }
        }
    
    def batch_predict(self, texts, model='logistic'):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts (list): List of texts to classify
            model (str): Which model to use
            
        Returns:
            list: List of predicted sentiments
        """
        return [self.predict(text, model) for text in texts]
    
    def save_model(self, filepath):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'nb_model': self.nb_model,
            'lr_model': self.lr_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load previously trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.nb_model = model_data['nb_model']
        self.lr_model = model_data['lr_model']
        self.is_trained = True


class TextPreprocessor:
    """Handles text cleaning and normalization."""
    
    def __init__(self):
        """Initialize preprocessor with NLTK resources."""
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            print("Downloading NLTK resources...")
            import nltk
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        """
        Clean and normalize text.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
