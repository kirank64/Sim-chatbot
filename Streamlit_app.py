import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (do this only once, before running the Streamlit app)
nltk.download('punkt')
nltk.download('stopwords') 

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Loads the saved model and vectorizer from the specified paths."""
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_path, "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Error: Could not find model or vectorizer files at {model_path} and {vectorizer_path}")
        return None, None

# Define default file paths (adjust as needed)
model_path = "best_model.pkl" 
vectorizer_path = "vectorizer.pkl" 

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

if model is None or vectorizer is None:
    st.stop()  # Stop the app if loading fails

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Chatbot response function
def chatbot_response(user_input):
    try:
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        intent = model.predict(vectorized_input)[0]
        responses = {
            "greeting": "Hi there! How can I assist you?",
            "joke": "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "booking": "Sure! Let me help you with that booking.",
            "weather": "Today's weather is sunny with a chance of awesomeness!",
            "music": "Playing your favorite tunes!",
            "help": "I'm here to help! What do you need?",
            "restaurants": "Looking for restaurants nearby...",
            "time": "It's time to shine! The current time is... (check your watch!)",
            "farewell": "Goodbye! Have a great day ahead!",
            "small_talk": "I'm just a chatbot, but I'm doing great. Thanks for asking!"
        }
        return responses.get(intent, "I didn't quite catch that. Could you try rephrasing?")
    except Exception as e: 
        st.error(f"An error occurred: {e}")
        return "I'm having some trouble right now. Please try again later."

# Streamlit app interface
st.title("Chatbot using NLP")
user_input = st.text_input("You:", placeholder="Type something...")
if user_input:
    st.write(f"Bot: {chatbot_response(user_input)}")
