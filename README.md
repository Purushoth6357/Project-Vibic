# Music Genre Classifier 🎵

An AI-powered music genre classification system using **Convolutional Neural Networks (CNN)**, **FastAPI**, and a modern web UI.

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135%2B-green)

## Features

✨ **AI-Powered Classification** – Upload any audio file and get instant genre predictions  
🎯 **10 Genres Supported** – Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock  
⚡ **Real-time Inference** – Fast predictions using keras/TensorFlow  
🎨 **Modern Web UI** – Beautiful, responsive frontend with Vanilla JavaScript  
🔌 **REST API** – Simple FastAPI endpoints for integration  
📊 **Feature Extraction** – Mel-spectrogram based audio analysis  

## Project Structure

```
Project-Vibic/
├── model.py              # CNN model definition & training logic
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
│
├── backend/
│   └── app.py           # FastAPI server
│
├── frontend/
│   ├── index.html       # Main UI
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
│
├── Data/
│   ├── features_*.csv   # Extracted audio features
│   ├── genres_original/ # Audio files by genre
│   └── images_original/ # Mel-spectrograms
│
└── models/
    ├── music_genre_model.keras  # Trained CNN model
    ├── norm_stats.npz           # Normalization statistics
    └── genre_labels.npy         # Genre label encoding
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Purushoth6357/Project-Vibic.git
cd Project-Vibic
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv music_env
music_env\Scripts\activate  # Windows
source music_env/bin/activate  # Linux/Mac

# Or using conda
conda create -n music_env python=3.10
conda activate music_env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application

**Start the backend server:**
```bash
cd backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Access the frontend:**
- Open browser: `http://localhost:8000`
- Upload an audio file (.mp3, .wav, .flac, etc.)
- Get instant genre prediction

## API Documentation

### Endpoints

#### 📤 POST `/predict`
Upload an audio file and get genre prediction.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@song.mp3"
```

**Response:**
```json
{
  "genre": "rock",
  "confidence": 0.95,
  "all_predictions": {
    "rock": 0.95,
    "metal": 0.03,
    "pop": 0.02,
    ...
  }
}
```

#### 📋 GET `/genres`
Get list of supported genres.

**Response:**
```json
{
  "genres": [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
  ]
}
```

#### 🏥 GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** Mel-spectrogram (128×128)
- **Features:** Librosa audio feature extraction
- **Sample Rate:** 22,050 Hz
- **Chunk Duration:** 3 seconds (with 1.5s overlap)
- **Framework:** TensorFlow/Keras

## Training Your Own Model

To retrain the model with custom data:

```bash
python train.py
```

This will:
1. Load audio files from `Data/genres_original/`
2. Extract Mel-spectrogram features
3. Train the CNN model
4. Save weights to `models/music_genre_model.keras`

## Deployment

### Option 1: Deploy on Render
```bash
# Install render CLI
npm install -g render

# Create render.yaml (already configured)
git push
```

### Option 2: Deploy on Railway
```bash
# Install railway CLI
npm install -g @railway/cli

# Deploy
railway up
```

### Option 3: Deploy Locally with Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run:
```bash
docker build -t music-classifier .
docker run -p 8000:8000 music-classifier
```

## Technologies Used

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) – Modern Python web framework
- **ML:** [TensorFlow/Keras](https://www.tensorflow.org/) – Neural networks
- **Audio:** [Librosa](https://librosa.org/) – Audio feature extraction
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Image Processing:** [PIL](https://pillow.readthedocs.io/)

## Performance

- **Model Accuracy:** ~85-92% (depends on training data quality)
- **Inference Time:** ~500ms per audio file
- **Supported Formats:** MP3, WAV, FLAC, OGG, M4A
- **Max File Size:** ~10MB

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Known Issues & Limitations

- Large audio files (>10MB) may take longer to process
- Best performance with music samples (may struggle with speech/noise)
- Model trained on GTZAN dataset; performance varies with different music styles

## Future Improvements

- [ ] Support for Spotify API integration
- [ ] Real-time audio stream classification
- [ ] User authentication & prediction history
- [ ] Model optimization for mobile deployment
- [ ] Multi-label genre classification
- [ ] Confidence score visualization

## Authors

- **Purushoth6357** – [GitHub](https://github.com/Purushoth6357)

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE) file for details.

## Acknowledgments

- GTZAN Music Genre Dataset
- TensorFlow & Keras teams
- FastAPI community

## Support

- 📧 Email: purushoth6357@gmail.com
- 🐛 Report issues: [GitHub Issues](https://github.com/Purushoth6357/Project-Vibic/issues)
- ⭐ Star the repo if it helps you!

---

**Made with ❤️ by Purushoth6357**
