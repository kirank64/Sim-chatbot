# --- Imports ---
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import RandomOverSampler  # To ensure class balance in subsets
import pickle

# --- Download NLTK resources ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Data Preparation ---
data = {
    'text': [
        "Hi there!", "Hello!", "Good morning", "Good evening", "Hey!",
        "Tell me something funny", "Make me laugh", "Do you know any jokes?", "Tell a joke", "I need a laugh",
        "I want to book a flight", "Reserve a table at a restaurant", "Can you book a cab for me?", "I need to book a hotel", "Schedule an appointment",
        "What's the weather like today?", "Tell me the weather forecast", "Is it going to rain today?", "What's the temperature outside?", "Check the weather in New York",
        "Play some music for me", "Can you play a song?", "I want to listen to something relaxing", "Start the playlist", "Play my favorite song",
        "Goodbye", "See you later", "Bye!", "Talk to you soon", "Catch you later"
    ],
    'intent': [
        "greeting", "greeting", "greeting", "greeting", "greeting",
        "joke", "joke", "joke", "joke", "joke",
        "booking", "booking", "booking", "booking", "booking",
        "weather", "weather", "weather", "weather", "weather",
        "music", "music", "music", "music", "music",
        "farewell", "farewell", "farewell", "farewell", "farewell"
    ]
}
df = pd.DataFrame(data)

# --- Text Preprocessing ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Remove empty rows
df = df[df['cleaned_text'] != ""].reset_index(drop=True)

# --- Feature Engineering ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['intent']

# --- Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Handle Class Imbalance with Oversampling ---
ros = RandomOverSampler(random_state=42)
X_train_dense, y_train = ros.fit_resample(X_train.toarray(), y_train)

# --- Model Training ---
svc_model = SVC(kernel='linear', probability=True)  # Define the base estimator
bagging_model = BaggingClassifier(
    estimator=svc_model,  # Pass SVC model as the estimator
    n_estimators=10,
    random_state=42
)

bagging_model.fit(X_train_dense, y_train)
y_pred = bagging_model.predict(X_test.toarray())

# --- Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging Model Accuracy: {accuracy:.3f}")

# --- Save the Model and Vectorizer ---
with open("bagging_model.pkl", "wb") as model_file:
    pickle.dump(bagging_model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
