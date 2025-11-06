import streamlit as st
import pickle
import os
import re
import gdown
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from dotenv import load_dotenv

load_dotenv()

# --- 1. NLTK Data Download (CLOUD SAFE) ---
@st.cache_resource
def download_nltk_data():
    """Downloads and configures NLTK data for cloud environments."""
    nltk_data_dir = '/tmp/nltk_data'
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
        print("✅ NLTK data downloaded and configured successfully.")
    except Exception as e:
        st.error(f"❌ Error downloading NLTK data: {e}")

download_nltk_data()

# --- 2. File Download Logic (Google Drive) ---
def download_file_from_google_drive(file_id, destination):
    """Downloads a file from a Google Drive shareable link using gdown."""
    
    if os.path.exists(destination):
        print(f"{destination} already exists. Skipping download.")
        return True

    with st.spinner(f'Downloading {destination}... (This may take a minute)'):
        try:
            # Construct the Google Drive URL
            url = f'https://drive.google.com/uc?id={file_id}'
            # Use gdown to download
            gdown.download(url, destination, quiet=False)
            
            if os.path.exists(destination):
                print(f"{destination} downloaded successfully.")
                return True
            else:
                st.error(f"Download failed for {destination}. File not found after gdown command.")
                return False
        except Exception as e:
            st.error(f"Error downloading {destination} from Google Drive: {e}")
            return False

# --- 3. Model Loading (MODIFIED for Google Drive) ---
MODEL_FILE_ID = os.environ.get('MODEL_FILE_ID')
VECTORIZER_FILE_ID = os.environ.get('VECTORIZER_FILE_ID')

MODEL_FILENAME = 'sentiment_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'

@st.cache_resource
def load_pipeline():
    """Downloads and loads the pre-trained model and vectorizer."""
    
    # Download files first
    model_ready = download_file_from_google_drive(MODEL_FILE_ID, MODEL_FILENAME)
    vectorizer_ready = download_file_from_google_drive(VECTORIZER_FILE_ID, VECTORIZER_FILENAME)

    if not model_ready or not vectorizer_ready:
        st.error("Model or vectorizer files could not be downloaded. Please check your Google Drive File IDs and share settings.")
        return None, None
        
    # Load objects from disk
    try:
        with open(MODEL_FILENAME, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(VECTORIZER_FILENAME, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, vectorizer = load_pipeline()

# --- 4. Preprocessing Function ---
# Initialize tools AFTER NLTK download
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return " ".join(cleaned_tokens)

# --- 5. Prediction Logic ---
def predict_sentiment(review_text):
    if model is None or vectorizer is None:
        return "Model Error", "0.00%"
    try:
        cleaned_review = preprocess_text(review_text)
        review_vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vector)[0]
        probability_scores = model.predict_proba(review_vector)[0]
        class_map = {cls: prob for cls, prob in zip(model.classes_, probability_scores)}
        confidence = class_map.get(prediction, max(probability_scores))
        return prediction, f"{confidence * 100:.2f}%"
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "ERROR", "0.00%"

# --- 6. Streamlit UI ---
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    /* ... (Your CSS styles remain exactly the same) ... */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        min-height: 95vh;
        display: flex;
        flex-direction: column;
    }
    .big-sentiment {
        font-size: 3rem; 
        font-weight: 800;
        margin-top: 0.5rem;
    }
    .sentiment-positive { color: #16a34a; }
    .sentiment-negative { color: #dc2626; }
    #textarea-input {
        min-height: 400px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a review on the left, and see the real-time NLP analysis on the right.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Enter Review")
    review_input = st.text_area(
        "Paste your movie review text below:",
        height=300,
        placeholder="e.g., 'A cinematic disaster! The worst writing and acting I have endured in years.'",
        key="textarea-input"
    )
    if st.button("Analyze Sentiment", type="primary"):
        if review_input:
            st.session_state['review'] = review_input 
            st.session_state['sentiment'], st.session_state['confidence'] = predict_sentiment(review_input)
            st.session_state['show_result'] = True
        else:
            st.warning("Please enter a review to analyze.")

with col2:
    st.subheader("2. Analysis Result")
    if st.session_state.get('show_result', False):
        sentiment = st.session_state['sentiment']
        confidence = st.session_state['confidence']
        is_positive = sentiment.lower() == 'positive'
        sentiment_class = "sentiment-positive" if is_positive else "sentiment-negative"
        sentiment_text = f"{sentiment.upper()} {'😄' if is_positive else '😟'}"
        st.markdown(f"**Predicted Emotion:**", unsafe_allow_html=True)
        st.markdown(f"<div class='big-sentiment {sentiment_class}'>{sentiment_text}</div>", unsafe_allow_html=True)
        st.metric("Confidence Score", confidence)
        st.info(f"**Review Analyzed:** \n\n{st.session_state['review']}")
    else:
        st.info("Results will appear here after you enter a review and click 'Analyze Sentiment'.")
        
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8501))
    # This runs the Streamlit app using the correct port on Render
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")