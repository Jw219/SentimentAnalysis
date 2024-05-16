import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import scipy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# Load the tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
model = AutoModelForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

# Load the dataset
df = pd.read_csv('data.csv')

# Drop duplicate sentences
df.drop_duplicates(subset=['Sentence'], keep='first', inplace=True)

# Convert sentences to a list
X = df['Sentence'].to_list()

# Extract true labels
y = df['Sentiment'].to_list()

# Initialize lists to store predictions and probabilities
preds = []
preds_proba = []

# Define tokenizer arguments
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}

# Iterate over each sentence
for x in tqdm(X):
    with torch.no_grad():
        # Tokenize the input sentence
        input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)

        # Get logits from the model
        outputs = model(**input_sequence)
        logits = outputs.logits

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()

        # Map probabilities to labels
        label_scores = {model.config.id2label[i]: probs[i] for i in range(len(probs))}

        # Get the sentiment with the highest score
        sentimentFinancialBERT = max(label_scores, key=label_scores.get)
        probabilityFinancialBERT = max(label_scores.values())

        # Append the results to the lists
        preds.append(sentimentFinancialBERT)
        preds_proba.append(probabilityFinancialBERT)

# Add predictions to the DataFrame
df['predict_FinancialBERT'] = preds

# Display the DataFrame
print(df)

# Evaluate the predictions
print(classification_report(y, preds))

# Compute the confusion matrix
cm = confusion_matrix(y, preds)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.config.id2label.values(), yticklabels=model.config.id2label.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()