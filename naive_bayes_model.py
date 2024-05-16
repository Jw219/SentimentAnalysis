

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import string
import warnings
warnings.filterwarnings("ignore")



data = pd.read_csv('data.csv')

data.drop_duplicates(inplace=True) # Drop duplicates

data.dropna(inplace=True) # Handle missing values (if any)

stop_words = set(stopwords.words('english')) # Text preprocessing
def preprocess_text(text): # Tokenize the text

    tokens = word_tokenize(text) # Convert to lowercase

    tokens = [word.lower() for word in tokens] # Remove punctuation

    tokens = [word for word in tokens if word not in string.punctuation] # Remove stopwords

    tokens = [word for word in tokens if word not in stop_words] # Join tokens back into string

    return " ".join(tokens)

# Apply preprocessing to the 'Sentence' column
data['Processed_Text'] = data['Sentence'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Processed_Text'])
y = data['Sentiment']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Create and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# The code was copied and then added onto to create the confussion matrix and the accuracy print out
# Appiah, E. K. (2022). "Sentiment Analysis using SVM, Naive Bayes & RF." Kaggle, Available online:
#  https://www.kaggle.com/code/emmanuelkwasiappiah/sentiment-analysis-using-svm-naive-bayes-rf (accessed April 20, 2024).