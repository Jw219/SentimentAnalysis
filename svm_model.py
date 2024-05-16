

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Import and download the required libraries and theyre attributes


# Load the dataset from the uploaded file area
data = pd.read_csv('data.csv')

# Define stopwords list
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):

  text = text.lower()  # Lowercase conversion
  text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
  text = " ".join([word for word in word_tokenize(text) if word not in stop_words])
  # Stemming
  stemmer = PorterStemmer()
  text = " ".join([stemmer.stem(word) for word in text.split()])
  return text

# Apply preprocessing to the 'Sentence' column
data['Processed_Text'] = data['Sentence'].apply(preprocess_text)

# Define features and target variable
X = data['Processed_Text']
y = data['Sentiment']

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=2000)
X_features = vectorizer.fit_transform(X)

# Split the train, test data at a 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = svm_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
# Create a confusion matrix based on the results
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()