"""
streamlit_app.py — Streamlit Frontend for Music Genre Classification

A simple web interface to upload audio files and predict their music genre.
Run with: streamlit run streamlit_app.py
"""

import os
import sys
import tempfile
import numpy as np
import librosa
from PIL import Image
import streamlit as st

# Add parent directory to path so we can import model.py
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model


# ─── Configuration ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'music_genre_model.keras')
NORM_STATS_PATH = os.path.join(MODEL_DIR, 'norm_stats.npz')
GENRE_LABELS_PATH = os.path.join(MODEL_DIR, 'genre_labels.npy')

SAMPLE_RATE = 22050
CHUNK_DURATION = 3
OVERLAP_DURATION = 1.5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SHAPE = (128, 128)

GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]


# ─── Streamlit Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Genre Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS for Dark Theme ──────────────────────────────────────
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background-color: #0a0a0f;
        color: #f0f0f5;
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0a0a0f;
    }
    
    .stTitle {
        color: #f0f0f5;
        font-weight: 700;
    }
    
    .stSubheader {
        color: #f0f0f5;
    }
    
    .stFileUploader {
        background-color: #12121a;
        border: 2px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 24px;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: transparent;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #a855f7, #06b6d4);
        color: #f0f0f5;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 24px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        box-shadow: 0 0 30px rgba(168, 85, 247, 0.4);
        transform: translateY(-2px);
    }
    
    .stMetric {
        background-color: #12121a;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px;
    }
    
    .stSuccess, .stInfo, .stError {
        background-color: #12121a;
        border-left: 4px solid #a855f7;
    }
</style>
""", unsafe_allow_html=True)


# ─── Caching Model Loading ──────────────────────────────────────────
@st.cache_resource
def load_model_and_stats():
    """Load the trained model and normalization stats (cached)."""
    genreModel = None
    normMean = None
    normStd = None

    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model as keras_load_model
        genreModel = keras_load_model(MODEL_PATH)
    else:
        st.error(f"Model not found at {MODEL_PATH}. Run train.py first.")
        return None, None, None

    if os.path.exists(NORM_STATS_PATH):
        stats = np.load(NORM_STATS_PATH)
        normMean = float(stats['mean'])
        normStd = float(stats['std'])
    else:
        st.warning(f"Normalization stats not found at {NORM_STATS_PATH}")

    if os.path.exists(GENRE_LABELS_PATH):
        loadedGenres = np.load(GENRE_LABELS_PATH, allow_pickle=True).tolist()

    return genreModel, normMean, normStd


# ─── Audio Processing ──────────────────────────────────────────────
def extract_mel_spectrogram(audioData, sampleRate):
    """Extract mel spectrogram from audio chunk."""
    spec = librosa.feature.melspectrogram(
        y=audioData, sr=sampleRate, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    specDb = librosa.power_to_db(spec, ref=np.max)

    specImage = Image.fromarray(specDb)
    specImage = specImage.resize((TARGET_SHAPE[1], TARGET_SHAPE[0]), Image.BILINEAR)
    return np.array(specImage)


def process_audio_for_prediction(filePath):
    """
    Process an audio file into mel spectrogram chunks for prediction.
    Same pipeline as training.
    """
    audioData, sr = librosa.load(filePath, sr=SAMPLE_RATE)

    chunkSamples = int(CHUNK_DURATION * SAMPLE_RATE)
    overlapSamples = int(OVERLAP_DURATION * SAMPLE_RATE)
    stepSamples = chunkSamples - overlapSamples

    spectrograms = []
    numChunks = max(1, (len(audioData) - chunkSamples) // stepSamples + 1)

    for i in range(numChunks):
        startSample = i * stepSamples
        endSample = startSample + chunkSamples
        chunk = audioData[startSample:endSample]

        if len(chunk) < chunkSamples:
            chunk = np.pad(chunk, (0, chunkSamples - len(chunk)), mode='constant')

        spec = extract_mel_spectrogram(chunk, SAMPLE_RATE)
        spectrograms.append(spec)

    return np.array(spectrograms)


# ─── Main UI ────────────────────────────────────────────────────────
def main():
    st.title("Genre Classifier")
    st.markdown("AI-Powered Music Analysis")

    # Load model
    genreModel, normMean, normStd = load_model_and_stats()

    if genreModel is None:
        st.stop()

    # File uploader
    st.subheader("Drop your audio file here")
    uploadedFile = st.file_uploader(
        "or click to browse",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        label_visibility="collapsed"
    )

    # Process uploaded file
    if uploadedFile is not None:
        # Display file info
        st.write(f"**File:** {uploadedFile.name}")
        st.write(f"**Size:** {uploadedFile.size / 1024:.1f} KB")

        # Predict button
        if st.button("Classify Genre", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploadedFile.name)[1]) as tmp:
                tmp.write(uploadedFile.getbuffer())
                tmpPath = tmp.name

            try:
                # Show processing status
                with st.spinner("Analyzing..."):
                    # Process audio
                    spectrograms = process_audio_for_prediction(tmpPath)

                    # Normalize using training stats
                    if normMean is not None and normStd is not None:
                        spectrograms = (spectrograms - normMean) / normStd

                    # Add channel dimension
                    spectrograms = spectrograms[..., np.newaxis]

                    # Predict on each chunk
                    predictions = genreModel.predict(spectrograms, verbose=0)

                # Average predictions across all chunks
                avgPrediction = np.mean(predictions, axis=0)

                predictedIdx = int(np.argmax(avgPrediction))
                predictedGenre = GENRES[predictedIdx]
                confidence = float(avgPrediction[predictedIdx])

                allScores = {
                    genre: float(score)
                    for genre, score in zip(GENRES, avgPrediction)
                }

                # Display results
                st.success("Prediction Complete!")

                # Main result
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Predicted Genre",
                        value=predictedGenre.upper()
                    )

                with col2:
                    st.metric(
                        label="Confidence",
                        value=f"{confidence * 100:.1f}%"
                    )

                # Scores table
                st.subheader("Genre Scores")
                scores_list = [
                    {"Genre": g.capitalize(), "Score": f"{allScores[g]:.4f}"}
                    for g in GENRES
                ]
                scores_list.sort(key=lambda x: float(x["Score"]), reverse=True)
                st.dataframe(scores_list, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

            finally:
                # Clean up temp file
                if os.path.exists(tmpPath):
                    os.unlink(tmpPath)


if __name__ == '__main__':
    main()
