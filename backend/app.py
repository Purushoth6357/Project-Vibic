"""
app.py — FastAPI Backend for Music Genre Classification

Endpoints:
    POST /predict   → Upload audio file, returns genre prediction
    GET  /genres    → Returns list of supported genres
    GET  /health    → Health check

Serves the frontend static files from ../frontend/
"""

import os
import sys
import tempfile
import numpy as np
import librosa
from PIL import Image
import webbrowser
import threading
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Add parent directory to path so we can import model.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import build_model


# ─── Configuration ──────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'music_genre_model.keras')
NORM_STATS_PATH = os.path.join(MODEL_DIR, 'norm_stats.npz')
GENRE_LABELS_PATH = os.path.join(MODEL_DIR, 'genre_labels.npy')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

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


# ─── App Setup ──────────────────────────────────────────────────────
app = FastAPI(
    title="Music Genre Classifier",
    description="Upload audio to predict its music genre using a CNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global State ───────────────────────────────────────────────────
genreModel = None
normMean = None
normStd = None


def load_model_and_stats():
    """Load the trained model and normalization stats on startup."""
    global genreModel, normMean, normStd

    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model as keras_load_model
        genreModel = keras_load_model(MODEL_PATH)
        print(f"  [OK] Model loaded from {MODEL_PATH}")
    else:
        print(f"  [!] Model not found at {MODEL_PATH}")
        print(f"    Run train.py first to train and save the model.")

    if os.path.exists(NORM_STATS_PATH):
        stats = np.load(NORM_STATS_PATH)
        normMean = float(stats['mean'])
        normStd = float(stats['std'])
        print(f"  [OK] Norm stats loaded: mean={normMean:.4f}, std={normStd:.4f}")
    else:
        print(f"  [!] Normalization stats not found at {NORM_STATS_PATH}")

    if os.path.exists(GENRE_LABELS_PATH):
        loadedGenres = np.load(GENRE_LABELS_PATH, allow_pickle=True).tolist()
        print(f"  [OK] Genre labels loaded: {loadedGenres}")


@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    print("\n" + "=" * 50)
    print("  Starting Music Genre Classifier API")
    print("=" * 50)
    load_model_and_stats()
    print("=" * 50 + "\n")
    
    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(2)  # Wait 2 seconds for server to be fully ready
        webbrowser.open("http://localhost:8000")
    
    thread = threading.Thread(target=open_browser, daemon=True)
    thread.start()


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


# ─── API Endpoints ──────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": genreModel is not None,
        "genres": GENRES
    }


@app.get("/genres")
async def get_genres():
    """Return list of supported genre classes."""
    return {"genres": GENRES, "count": len(GENRES)}


@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    """
    Upload an audio file and get genre prediction.

    Returns:
        genre:      Predicted genre name
        confidence: Confidence score (0-1)
        all_scores: Dict of all genre scores
    """
    if genreModel is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first to train the model."
        )

    # Validate file type
    allowedExtensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    fileExt = os.path.splitext(file.filename)[1].lower()
    if fileExt not in allowedExtensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {fileExt}. Allowed: {allowedExtensions}"
        )

    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=fileExt) as tmp:
            content = await file.read()
            tmp.write(content)
            tmpPath = tmp.name

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
            genre: round(float(score), 4)
            for genre, score in zip(GENRES, avgPrediction)
        }

        return {
            "genre": predictedGenre,
            "confidence": round(confidence, 4),
            "all_scores": allScores,
            "chunks_analyzed": len(spectrograms),
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Clean up temp file
        if os.path.exists(tmpPath):
            os.unlink(tmpPath)


# ─── Serve Frontend ────────────────────────────────────────────────
# Serve static files (CSS, JS) directly from /static path
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend index.html"""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # Mount static folder to serve CSS, JS, and other assets
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─── Run Server ─────────────────────────────────────────────────────
if __name__ == '__main__':
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
