import streamlit as st
import tensorflow as tf
from gensim.models import KeyedVectors
import spacy
import numpy as np

# --- Load Required Models and Resources ---

# Load the trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('prediction.keras')

# Load Word2Vec embeddings
@st.cache_resource
def load_word2vec():
    return KeyedVectors.load("word2vec.model")

# Load Spacy tokenizer
@st.cache_resource
def load_spacy():
    return spacy.load("spacy_model")

# --- Text Preprocessing and Prediction ---

def preprocess_and_predict(description, model, word2vec, nlp):
    # Tokenize description
    tokens = [token.text.lower() for token in nlp(description) if token.is_alpha]
    # Create mean vector using Word2Vec
    vectors = [word2vec[word] for word in tokens if word in word2vec]
    if vectors:
        mean_vector = np.mean(vectors, axis=0)
    else:
        mean_vector = np.zeros(word2vec.vector_size)
    # Predict using the loaded model
    prediction = model.predict(mean_vector.reshape(1, -1))[0][0]
    genre = "Horror" if prediction > 0.5 else "Romance"
    confidence = prediction if genre == "Horror" else 1 - prediction
    return genre, confidence

# --- Streamlit App ---

st.title("Movie Genre Prediction App")
st.write("Enter a movie description to predict whether it's Horror or Romance.")

# Input field for user to enter a movie description
user_input = st.text_area("Enter movie description:")

# Load resources
model = load_model()
word2vec = load_word2vec()
nlp = load_spacy()

if st.button("Predict"):
    if user_input.strip():
        # Make prediction
        genre, confidence = preprocess_and_predict(user_input, model, word2vec, nlp)
        # Display result
        st.write(f"### Predicted Genre: {genre}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter a movie description.")
