# 🧬 DiabetesAI — Clinical Classification System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An AI-powered diabetes classification system that combines structured clinical data with NLP analysis of medical notes to predict diabetes type (Type 1, Type 2, Gestational, or Other).

![DiabetesAI Interface](https://via.placeholder.com/1200x600/0f1117/4f8ef7?text=DiabetesAI+Clinical+Interface)

---

## 🎯 Features

- **Multi-Modal Classification**: Combines structured features (age, BMI, glucose, insulin) with clinical note analysis
- **Ensemble Learning**: Integrates LogisticRegression baseline with Bio_ClinicalBERT fine-tuning
- **Professional Web UI**: Dark-themed medical interface with real-time predictions and confidence scores
- **Property-Based Testing**: Rigorous correctness validation using Hypothesis framework
- **MLOps Pipeline**: DVC-managed data versioning and MLflow experiment tracking
- **Production-Ready API**: FastAPI backend with health checks and error handling

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                          │
│              (Professional Clinical UI)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│              POST /predict  •  GET /health                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Ensemble Classifier                        │
│   ┌──────────────────────┐   ┌──────────────────────┐       │
│   │  Structured Model    │   │   BERT Classifier    │       │
│   │  (LogisticRegression)│   │ (Bio_ClinicalBERT)   │       │
│   └──────────────────────┘   └──────────────────────┘       │
│              Weighted Probability Fusion                    │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing Pipeline                       │
│   Data Ingestion → NLP Preprocessing → Feature Extraction   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/forex911/diabetes-type-classifier.git
   cd diabetes-type-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Train the model**
   ```bash
   python -m diabetes_identifier.models.train
   ```
   This generates synthetic EHR data and trains the baseline classifier (~2-3 minutes).

4. **Start the web application**
   ```bash
   uvicorn diabetes_identifier.api.main:app --reload --port 8000
   ```

5. **Open your browser**
   ```
   http://localhost:8000
   ```

---

## 📊 Usage

### Web Interface

The professional UI provides:
- **Dual input controls**: Sliders + number inputs with live sync
- **Clinical reference ranges**: Normal values displayed for each metric
- **Real-time predictions**: Instant classification with confidence scores
- **Probability visualization**: Animated bars showing all class probabilities
- **Input summary**: Echo of submitted patient data

### API Endpoints

#### `POST /predict`
Classify a patient based on clinical measurements.

**Request:**
```json
{
  "age": 45,
  "bmi": 28.5,
  "glucose": 180,
  "insulin": 20,
  "notes": "Patient presents with polyuria and polydipsia..."
}
```

**Response:**
```json
{
  "label": 1,
  "diagnosis": "Type 2",
  "probabilities": {
    "Type 1": 0.1234,
    "Type 2": 0.6789,
    "Gestational": 0.0987,
    "Other": 0.0990
  }
}
```

#### `GET /health`
Check API status.

**Response:**
```json
{
  "status": "ok"
}
```

---

## 🧪 Testing

Run the full test suite:
```bash
pytest diabetes_identifier/tests/
```

Run property-based tests only:
```bash
pytest diabetes_identifier/tests/test_property_*.py
```

Property-based tests validate:
- Data loader correctness (no data loss, type preservation)
- NLP preprocessing invariants (PHI masking, tokenization)
- BERT tokenizer properties (reversibility, length constraints)

---

## 📁 Project Structure

```
diabetes-classifier/
├── diabetes_identifier/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API routes and model loading
│   │   └── static/            # Frontend assets
│   │       └── index.html     # Professional web UI
│   ├── data/                  # Data storage
│   │   ├── raw/               # Synthetic EHR data
│   │   ├── processed/         # Preprocessed features
│   │   └── annotations/       # Manual labels (if any)
│   ├── models/                # Trained model artifacts
│   │   ├── train.py           # Training pipeline
│   │   └── evaluate.py        # Model evaluation
│   ├── nlp/                   # NLP processing
│   │   ├── preprocessing.py   # Text cleaning, PHI masking
│   │   └── embedding.py       # BERT/TF-IDF embeddings
│   ├── utils/                 # Utilities
│   │   ├── config.py          # Configuration dataclasses
│   │   ├── data_loader.py     # CSV ingestion
│   │   ├── logger.py          # Structured logging
│   │   └── generate_mock_data.py  # Synthetic data generation
│   └── tests/                 # Test suite
│       ├── test_property_*.py # Property-based tests
│       └── test_*.py          # Unit tests
├── dvc.yaml                   # DVC pipeline definition
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🔬 Model Details

### Baseline Models
- **LogisticRegression**: Trained on structured features (age, BMI, glucose, insulin)
- **DecisionTreeClassifier**: Alternative baseline for comparison

### BERT Fine-Tuning
- **Model**: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- **Task**: 4-class sequence classification
- **Input**: Clinical notes with PHI masking
- **Training**: HuggingFace Trainer with 1 epoch (configurable)

### Ensemble
- Weighted probability fusion: `P_final = w1 * P_structured + w2 * P_bert`
- Default weights: `[0.5, 0.5]` (configurable in `config.py`)

---

## 🛠️ Configuration

Edit `diabetes_identifier/utils/config.py` to customize:

```python
@dataclass
class Config:
    ingestion: IngestionConfig       # CSV path, missing value strategy
    preprocessing: PreprocConfig     # Outlier clipping, SMOTE threshold
    nlp: NLPConfig                   # PHI tokens, default notes text
    embedding: EmbeddingConfig       # "bert" or "tfidf" mode
    training: TrainingConfig         # Random seed, ensemble weights
    evaluation: EvaluationConfig     # Min F1 threshold, CV folds
    api: APIConfig                   # Host, port, log level
    mlflow: MLflowConfig             # Tracking URI, experiment name
```

---

## 📈 MLOps Pipeline

### DVC Stages

```bash
# Generate synthetic data
dvc repro prepare_data

# Preprocess and extract features
dvc repro preprocess

# Train models
dvc repro train
```

### MLflow Tracking

Experiments are automatically logged to `mlruns/`:
- Model hyperparameters
- Training metrics
- Artifact paths

View the MLflow UI:
```bash
mlflow ui
```

---

## 🎨 Frontend Preview

The web interface features:
- **Dark medical-grade theme** with gradient accents
- **Live API status indicator** with pulse animation
- **Dual-input controls** (sliders + number fields)
- **Clinical reference ranges** for each measurement
- **Animated probability bars** with color-coded diabetes types
- **Confidence badges** (High / Moderate / Low)
- **Responsive design** for mobile and desktop

---

## 🧩 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **ML/NLP** | scikit-learn, transformers, spaCy, PyTorch |
| **Data** | pandas, numpy, imbalanced-learn |
| **MLOps** | DVC, MLflow |
| **Testing** | pytest, Hypothesis |
| **Frontend** | Vanilla JS, CSS3 (no frameworks) |

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**For research and educational purposes only.** This system is not a substitute for professional medical diagnosis. Always consult qualified healthcare providers for clinical decisions.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- **Bio_ClinicalBERT** by Emily Alsentzer et al.
- **FastAPI** framework by Sebastián Ramírez
- **Hypothesis** property-based testing library

---

<div align="center">
  <sub>Built with ❤️ for advancing clinical AI research</sub>
</div>
