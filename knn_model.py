import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset file 
data = pd.read_csv('stock_13k.csv', encoding='latin1')


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




# Create the train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.2, random_state=42)





# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train_tfidf, y_train)
knn_score = knn.score(X_test_tfidf, y_test)
print("Results for KNN Classifier with tfidf")
print(knn_score)
y_pred_knn = knn.predict(X_test_tfidf)



cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix:\n", cm_knn)  # Print the full confusion matrix


conf_matrix = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
