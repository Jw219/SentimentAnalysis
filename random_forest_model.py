

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
#Import the needed libraries and their entitities

nltk.download('stopwords')

# Load the data from the file area
data = pd.read_csv('data.csv')

# Preprocessing function
def preprocess_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', str(text))
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)
    # Converting to Lowercase
    text = text.lower()
    return text

# Apply preprocessing
data['Processed_Sentence'] = data['Sentence'].apply(preprocess_text)

# Split data into features and labels
X = data['Processed_Sentence']
y = data['Sentiment']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=4500, min_df=6, max_df=0.8, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(X).toarray()

# Split the data into train and test at a ratio of 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# call on the RandomForestClassifier to complete classification on the dataset
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

# Predictions
predictions = text_classifier.predict(X_test)

# Evaluation of the model using the 20% of the dataset that wasnt used for training
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

# Compute the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot the confusion matrix
plt.figure(figsize=(9, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# Code for the Random Forest model was gathered from this source
# Stack Abuse. (No date). "Python for NLP: Sentiment Analysis with Scikit-Learn."
# Stack Abuse, Available online: https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/ (accessed April 20, 2024).
# The code was used as the basis for the model but then edited for the use of the Financial Analysis dataset and to create the confusion matric
# and to try and increase the accuracy by fine tuining the max_features of the TF-IDF vectorisation and the n_estimators