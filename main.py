
"""
Main script to run weather sentiment analysis demonstration.

This script:
1. Loads sample weather-related comments
2. Trains sentiment analysis models
3. Evaluates model performance
4. Makes predictions on new weather-related text
5. Generates comparison visualizations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_analyzer import SentimentAnalyzer
import warnings

warnings.filterwarnings('ignore')



def load_sample_data():
    """Load sample weather sentiment data from CSV."""
    import os
    data_path = os.path.join('data', 'weather_sentiment_samples.csv')
    df = pd.read_csv(data_path)
    return df['text'].tolist(), df['label'].tolist()


def prepare_data():
    """Generate or load training data."""
    texts, labels = load_sample_data()
    return train_test_split(texts, labels, test_size=0.2, random_state=42)


def evaluate_models(analyzer, X_test, y_test):
    """Evaluate all three models on test data."""
    
    results = {
        'Logistic Regression': [],
        'Naive Bayes': [],
        'VADER': []
    }
    
    # Logistic Regression predictions
    lr_preds = [analyzer.predict(text, model='logistic') for text in X_test]
    results['Logistic Regression'] = lr_preds
    
    # Naive Bayes predictions
    nb_preds = [analyzer.predict(text, model='naive_bayes') for text in X_test]
    results['Naive Bayes'] = nb_preds
    
    # VADER predictions
    vader_preds = [analyzer.predict(text, model='vader') for text in X_test]
    results['VADER'] = vader_preds
    
    # Calculate accuracies
    accuracies = {}
    for model_name, preds in results.items():
        accuracy = accuracy_score(y_test, preds)
        accuracies[model_name] = accuracy
        print(f"\n{model_name} Accuracy: {accuracy:.2%}")
        print(f"Classification Report:\n{classification_report(y_test, preds)}")
    
    return results, accuracies, y_test


def plot_accuracy_comparison(accuracies):
    """Create bar chart comparing model accuracies."""
    
    models = list(accuracies.keys())
    accuracies_list = list(accuracies.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies_list, color=['#2ecc71', '#3498db', '#e74c3c'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Sentiment Analysis Model Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: results/model_accuracy_comparison.png")
    plt.close()


def plot_confusion_matrix(y_test, predictions, model_name):
    """Create confusion matrix visualization."""
    
    labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(y_test, predictions, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    filename = f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()



def demonstrate_predictions(analyzer):
    """Show example predictions on new weather-related text."""
    
    test_samples = [
        "The sun is shining and I feel great!",
        "This rain is making me so sad.",
        "It's just an average day outside.",
    ]
    
    print("\n" + "="*70)
    print("SAMPLE WEATHER SENTIMENT PREDICTIONS")
    print("="*70)
    
    for text in test_samples:
        print(f"\nText: {text}")
        
        # Get predictions from all models
        lr_pred = analyzer.predict(text, model='logistic')
        nb_pred = analyzer.predict(text, model='naive_bayes')
        vader_pred = analyzer.predict(text, model='vader')
        
        # Get probabilities from logistic regression
        lr_scores = analyzer.predict_with_scores(text, model='logistic')
        
        print(f"  Logistic Regression: {lr_pred}")
        print(f"  Naive Bayes: {nb_pred}")
        print(f"  VADER: {vader_pred}")
        print(f"  Confidence Scores: {lr_scores['scores']}")


def main():
    """Main execution function."""
    
    print("="*70)
    print("SENTIMENT ANALYSIS PROJECT")
    print("="*70)
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Load and prepare data
    print("\n[1/5] Loading data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"✓ Loaded {len(X_train)} training and {len(X_test)} test samples")
    
    # Initialize analyzer
    print("\n[2/5] Initializing sentiment analyzer...")
    analyzer = SentimentAnalyzer()
    print("✓ Analyzer initialized")
    
    # Train models
    print("\n[3/5] Training models...")
    training_results = analyzer.train(X_train, y_train)
    print(f"✓ Training complete")
    print(f"  - Naive Bayes Accuracy: {training_results['nb_accuracy']:.2%}")
    print(f"  - Logistic Regression Accuracy: {training_results['lr_accuracy']:.2%}")
    
    # Evaluate models
    print("\n[4/5] Evaluating models on test data...")
    results, accuracies, y_test_val = evaluate_models(analyzer, X_test, y_test)
    
    # Visualizations
    print("\n[5/5] Generating visualizations...")
    plot_accuracy_comparison(accuracies)
    
    # Plot confusion matrices for best model
    best_model = max(accuracies, key=accuracies.get)
    lr_preds = [analyzer.predict(text, model='logistic') for text in X_test]
    plot_confusion_matrix(y_test, lr_preds, 'Logistic Regression')
    
    # Demonstrate predictions
    demonstrate_predictions(analyzer)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best Model: {best_model}")
    print(f"Best Accuracy: {accuracies[best_model]:.2%}")
    print("\nVisualizations saved to: results/")
    print("="*70)


if __name__ == "__main__":
    main()
