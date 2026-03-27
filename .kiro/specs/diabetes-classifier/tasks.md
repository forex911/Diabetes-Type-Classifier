# Implementation Plan: Diabetes Classifier

## Overview

Incremental build-out of the end-to-end diabetes classification system: project scaffold → data layer → NLP/embedding → model training → evaluation/interpretability → FastAPI → Docker → tests → MLflow/DVC → bonus features.

## Tasks

- [x] 1. Project scaffold and configuration
  - Create the directory tree: `diabetes_identifier/{data/{raw,processed,annotations},nlp,models,pipelines,api,utils,experiments,tests}`
  - Implement `utils/config.py` with the `Config` dataclass hierarchy (`IngestionConfig`, `PreprocConfig`, `NLPConfig`, `EmbeddingConfig`, `TrainingConfig`, `EvaluationConfig`, `APIConfig`, `MLflowConfig`) and a `load_config` helper that reads a YAML/JSON file
  - Implement `utils/logger.py` with a structured JSON logger (stdlib `logging` + `python-json-logger`) and the `ExperimentTracker` MLflow wrapper stub (methods: `start_run`, `log_params`, `log_metrics`, `log_artifact`, `register_model`)
  - Create `requirements.txt` with pinned versions for all dependencies (pandas, numpy, scikit-learn, imbalanced-learn, spacy, transformers, torch, fastapi, uvicorn, mlflow, dvc, shap, pydantic, pytest, httpx, python-jose)
  - _Requirements: 10.3_

- [x] 2. Data ingestion and synthetic dataset
  - [x] 2.1 Implement `utils/data_loader.py` — `DataLoader.load(path)` that reads CSV into a DataFrame, raises `FileNotFoundError` with the path in the message when file is absent, logs and skips malformed rows, and imputes/flags missing structured fields (`age`, `bmi`, `glucose`, `insulin`) according to `ingestion.missing_strategy` (`mean`, `median`, `drop`)
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - [x] 2.2 Write property test for `DataLoader.load` round-trip
    - **Property 1: No data loss on valid CSV** — for any DataFrame with no malformed rows, `load(path)` returns a DataFrame with the same number of rows
    - **Validates: Requirements 1.1**
  - [x] 2.3 Generate `data/raw/ehr_synthetic.csv` — a script (`utils/generate_mock_data.py`) that produces ≥ 500 synthetic patient records covering all four diabetes classes with realistic distributions of age, BMI, glucose, insulin, and a `notes` column
    - _Requirements: 1.5_

- [x] 3. Structured data preprocessing
  - [x] 3.1 Implement `StructuredPreprocessor` in `nlp/preprocessing.py` with `fit`, `transform`, `fit_transform`, `serialize`, and `deserialize` methods
    - Normalize `[age, bmi, glucose, insulin]` to zero mean / unit variance using train-split statistics only
    - Clip values outside ±3σ to the boundary
    - Encode target label to `int ∈ [0, 3]` via `LabelEncoder` with the fixed `LABEL_MAP`
    - Apply SMOTE when any class represents < 15% of training samples
    - Pass inverse-frequency class weights to the classifier when `class_weights` is configured
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 3.2 Write property test for preprocessing round-trip
    - **Property 2: Serialization round-trip** — for any valid structured feature record, `serialize` then `deserialize` then `transform` produces byte-identical output to `transform` without serialization
    - **Validates: Requirements 9.1, 9.2**
  - [x] 3.3 Write unit tests for `StructuredPreprocessor`
    - Test missing value handling (mean/median/drop strategies)
    - Test ±3σ clipping boundary values
    - Test label encoding consistency
    - Test SMOTE activation when a class is < 15%
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 11.1_

- [x] 4. NLP preprocessing
  - [x] 4.1 Implement `NLPPreprocessor` in `nlp/preprocessing.py` with `preprocess` and `extract_entities` methods
    - Lowercase all text
    - Replace PHI placeholder tokens (`[NAME]`, `[DOB]`, `[ID]`, etc.) with `<PHI>`
    - Segment into sentences via spaCy `sentencizer`
    - Substitute empty/null notes with `nlp.default_notes_text` and log a warning
    - Extract NER entities (symptoms, medications, lab values) via spaCy model; return as `EntityMetadata` dicts
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [x] 4.2 Write unit tests for `NLPPreprocessor`
    - Test PHI replacement for each token type
    - Test empty/null note substitution and warning log
    - Test sentence segmentation produces a list
    - _Requirements: 3.1, 3.2, 3.4, 11.1_

- [x] 5. Feature extraction and embedding
  - [x] 5.1 Implement `FeatureExtractor` in `nlp/embedding.py` with `fit`, `transform`, `serialize`, and `deserialize` methods
    - `mode="tfidf"`: fit/transform TF-IDF with `max_features=10_000`
    - `mode="bert"`: mean-pool last hidden layer of `emilyalsentzer/Bio_ClinicalBERT` → 768-dim vector
    - Concatenate normalized structured vector + text embedding → combined feature vector
    - Serialize fitted TF-IDF vectorizer (`tfidf_vectorizer.pkl`) and BERT tokenizer config (`bert_tokenizer_config.json`) to `models/`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - [x] 5.2 Write property test for BERT tokenizer round-trip
    - **Property 3: BERT tokenizer encode→decode round-trip** — for any non-empty `notes` string, encoding then decoding recovers the original token sequence
    - **Validates: Requirements 9.3**
  - [x] 5.3 Write unit tests for `FeatureExtractor`
    - Test TF-IDF output dimension ≤ 10,000
    - Test BERT output dimension == 768
    - Test combined vector shape equals structured_dim + embedding_dim
    - Test mode switching between `"tfidf"` and `"bert"`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 6. Checkpoint — ensure data pipeline tests pass
  - Run `pytest tests/` covering tasks 2–5; ensure zero failures before proceeding to model training.

- [x] 7. Model training
  - [x] 7.1 Implement `ModelTrainer` in `models/train.py` with `train_baseline`, `train_bert`, and `train_ensemble` methods
    - `train_baseline`: trains `LogisticRegression` and `DecisionTreeClassifier` on combined feature vectors; uses fixed `random_seed`
    - `train_bert`: fine-tunes `Bio_ClinicalBERT` for 4-class classification via HuggingFace `Trainer`; uses fixed `random_seed`
    - `train_ensemble`: combines probability outputs from structured and BERT classifiers with configurable `ensemble_weights`; serializes `ensemble_weights.json`
    - Serialize all trained artifacts to `models/` (`structured_classifier.pkl`, `bert_classifier/`, `ensemble_weights.json`)
    - _Requirements: 5.1, 5.2, 5.3, 5.6_
  - [x] 7.2 Integrate `ExperimentTracker` into `ModelTrainer`
    - On run start: create MLflow run under `"diabetes-classifier"` experiment, log run ID to stdout
    - Log params: `model_type`, `embedding_mode`, `random_seed`, `n_train_samples`, all hyperparameters
    - _Requirements: 5.4, 12.1, 12.2_
  - [x] 7.3 Add DVC tracking for datasets and model artifacts
    - Add `data/raw/ehr_synthetic.csv.dvc` and `models/.dvc` entries
    - Commit `.dvc` metadata files to the repository
    - _Requirements: 5.5_

- [ ] 8. Model evaluation
  - [-] 8.1 Implement `Evaluator` in `models/evaluate.py` with an `evaluate(model, X, y, model_name)` method
    - Stratified 5-fold CV preserving class distribution
    - Compute macro F1, weighted F1, multi-class ROC-AUC (OvR), per-class precision-recall, confusion matrix, calibration curve
    - Report per-class recall separately for Type 1 and Gestational classes
    - Log warning when macro F1 < 0.70
    - Save confusion matrix, ROC curves, calibration curve as PNG to `experiments/{model_name}_{timestamp}/`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ] 8.2 Wire `Evaluator` metrics into `ExperimentTracker`
    - Log macro F1, weighted F1, ROC-AUC, per-class recall as MLflow metrics at end of each fold
    - Register best model (highest macro F1) in MLflow Model Registry as `"diabetes-classifier-production"`
    - _Requirements: 12.3, 12.4_
  - [ ] 8.3 Write unit tests for `Evaluator`
    - Test that evaluation returns all required metric keys
    - Test that PNG files are created in `experiments/`
    - Test that warning is logged when macro F1 < 0.70
    - _Requirements: 6.2, 6.4, 6.5_

- [ ] 9. Model interpretability
  - [ ] 9.1 Implement `InferenceEngine.explain` in `models/inference.py`
    - For structured classifier: compute SHAP values, return top 5 features with scores
    - For `DecisionTreeClassifier`: return decision path as ordered list of feature-threshold conditions
    - For BERT classifier: extract top 5 attention-weighted tokens from final attention layer
    - Return explanation as structured dict under `"explanation"` key
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [ ] 9.2 Write unit tests for `InferenceEngine.explain`
    - Test SHAP output contains exactly 5 items with `feature` and `score` keys
    - Test decision path output is a non-empty ordered list
    - Test BERT attention output contains exactly 5 token entries
    - _Requirements: 7.1, 7.2, 7.3_

- [ ] 10. Inference engine and API schemas
  - [ ] 10.1 Implement `InferenceEngine.predict` in `models/inference.py`
    - Load serialized artifacts on startup; raise descriptive error identifying missing artifact path if any artifact is absent
    - Run full inference pipeline: preprocess → embed → structured predict + BERT predict → ensemble → explain
    - Return `PredictionResult` with `prediction` (string label), `confidence` (float), `explanation` (dict)
    - _Requirements: 8.2, 9.4_
  - [ ] 10.2 Implement Pydantic schemas in `api/schemas.py`
    - `PatientRecord`, `ExplanationItem`, `PredictionResponse`, `HealthResponse`, `DriftReport`
    - _Requirements: 8.1, 8.2_
  - [ ] 10.3 Write unit tests for `InferenceEngine.predict`
    - Test output schema: `prediction` is a string in `LABEL_MAP.values()`, `confidence` ∈ [0.0, 1.0], `explanation` is a dict
    - Test that missing artifact raises descriptive error
    - _Requirements: 8.2, 9.4, 11.2_

- [ ] 11. FastAPI application
  - [ ] 11.1 Implement `api/main.py` with `POST /predict`, `GET /health` endpoints
    - `POST /predict`: validate `PatientRecord`, call `InferenceEngine.predict`, return `PredictionResponse`; return HTTP 422 on missing fields, HTTP 500 on internal errors with stack trace logged
    - `GET /health`: return `{"status": "ok"}` with HTTP 200 when model is loaded
    - Log each request: input fields (excluding raw notes), predicted label, confidence, latency in ms as structured JSON
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  - [ ] 11.2 Write integration tests for API endpoints
    - Test `POST /predict` with valid payload returns 200 with correct schema
    - Test `POST /predict` with missing required field returns 422
    - Test `GET /health` returns 200 `{"status": "ok"}`
    - _Requirements: 8.1, 8.2, 8.3, 8.5, 11.3_

- [ ] 12. Checkpoint — ensure all model and API tests pass
  - Run `pytest tests/`; ensure zero failures before proceeding to deployment and bonus features.

- [ ] 13. Training and inference pipelines
  - [ ] 13.1 Implement `pipelines/training_pipeline.py` — orchestrates `DataLoader → StructuredPreprocessor → NLPPreprocessor → FeatureExtractor → ModelTrainer → Evaluator → ExperimentTracker` end-to-end; accepts a `Config` object
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 12.1_
  - [ ] 13.2 Implement `pipelines/inference_pipeline.py` — orchestrates artifact loading → preprocessing → embedding → `InferenceEngine.predict` for a single `PatientRecord`; used by the API
    - _Requirements: 8.2_

- [ ] 14. Docker deployment
  - Implement `Dockerfile`: multi-stage build, install all dependencies from `requirements.txt`, copy model artifacts, expose port 8000, set `CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]`
  - Ensure `GET /health` returns HTTP 200 after container startup
  - _Requirements: 10.1, 10.2, 10.4_

- [ ] 15. Drift detection (bonus)
  - [ ] 15.1 Implement `DriftDetector.compute_psi` in `pipelines/inference_pipeline.py`
    - Compute PSI per structured numeric feature between training reference distribution and inference batch
    - Log warning with feature name and PSI value when PSI > 0.2
    - _Requirements: 13.1, 13.2_
  - [ ] 15.2 Add `POST /drift-report` endpoint to `api/main.py`
    - Accept a batch of records, call `DriftDetector.compute_psi`, return `DriftReport` JSON
    - _Requirements: 13.3_
  - [ ] 15.3 Write unit tests for `DriftDetector`
    - Test PSI is 0.0 when reference and current distributions are identical
    - Test warning is logged when PSI > 0.2
    - _Requirements: 13.1, 13.2_

- [ ] 16. Role-based access control (bonus)
  - [ ] 16.1 Add JWT bearer token middleware to `api/main.py`
    - Require valid JWT on all endpoints except `GET /health`; return HTTP 401 on missing/invalid token
    - Permit `admin` role access to `/drift-report` and retraining endpoints
    - Permit `doctor` role access to `POST /predict` only; return HTTP 403 on restricted endpoints
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  - [ ] 16.2 Write unit tests for RBAC middleware
    - Test unauthenticated request returns 401
    - Test `doctor` role can access `/predict` but gets 403 on `/drift-report`
    - Test `admin` role can access `/drift-report`
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 17. Final checkpoint — full test suite
  - Run `pytest tests/`; ensure zero failures across all unit and integration tests.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints (tasks 6, 12, 17) ensure incremental validation
- Property tests validate universal correctness properties; unit tests validate specific examples and edge cases
- Bonus tasks (15, 16) implement drift detection and RBAC and can be deferred
