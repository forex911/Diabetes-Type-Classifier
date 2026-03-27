# Requirements Document

## Introduction

An end-to-end machine learning system that classifies diabetes type (Type 1, Type 2, Gestational, Other) from structured EHR data (age, BMI, glucose, insulin) and unstructured clinical notes. The system ingests and preprocesses data, extracts NLP features via spaCy and Clinical BERT, trains interpretable and deep learning models, evaluates with medical-grade metrics, exposes predictions via a FastAPI REST API, and is packaged for Docker deployment.

## Glossary

- **System**: The diabetes-classifier application as a whole
- **Pipeline**: An ordered sequence of data transformation and model inference steps
- **EHR**: Electronic Health Record — structured patient data (age, BMI, lab values, etc.)
- **Clinical_Notes**: Free-text physician observations stored as unstructured strings
- **NLP_Pipeline**: The component responsible for preprocessing and embedding Clinical_Notes
- **Feature_Extractor**: The component that combines structured EHR features with NLP embeddings
- **Classifier**: The ML model that predicts diabetes type from extracted features
- **Ensemble**: A model that combines outputs from the structured Classifier and the BERT_Classifier
- **BERT_Classifier**: A fine-tuned ClinicalBERT model for multi-class diabetes classification
- **Inference_API**: The FastAPI application exposing the `/predict` endpoint
- **Experiment_Tracker**: The MLflow component that logs parameters, metrics, and artifacts
- **Data_Store**: The local filesystem directories (`data/raw/`, `data/processed/`, `data/annotations/`)
- **PHI**: Protected Health Information — identifiers that must be removed before processing
- **SMOTE**: Synthetic Minority Over-sampling Technique for handling class imbalance
- **SHAP**: SHapley Additive exPlanations — a framework for model interpretability
- **DVC**: Data Version Control — tool for versioning datasets and model artifacts
- **Drift_Detector**: The component that monitors incoming data distributions against training distributions

---

## Requirements

### Requirement 1: Data Ingestion

**User Story:** As a data engineer, I want to ingest structured EHR records and unstructured clinical notes from CSV files, so that the system has a unified dataset ready for preprocessing.

#### Acceptance Criteria

1. WHEN a CSV file is provided at a configured path, THE System SHALL load it into a Pandas DataFrame without data loss.
2. WHEN a CSV file contains missing values in structured fields (age, BMI, glucose, insulin), THE System SHALL impute or flag each missing value according to a configurable strategy (mean, median, or drop).
3. WHEN a CSV file is not found at the configured path, THE System SHALL raise a descriptive `FileNotFoundError` with the missing path included in the message.
4. WHEN a CSV file contains malformed rows, THE System SHALL log each malformed row index and skip those rows, continuing ingestion of valid rows.
5. THE System SHALL support a mock EHR dataset of at least 500 synthetic patient records covering all four diabetes classes (Type 1, Type 2, Gestational, Other).

---

### Requirement 2: Structured Data Preprocessing

**User Story:** As a data scientist, I want structured EHR fields cleaned and normalized, so that models receive consistent numeric inputs.

#### Acceptance Criteria

1. THE System SHALL normalize numeric fields (age, BMI, glucose, insulin) to zero mean and unit variance using statistics computed on the training split only.
2. WHEN a numeric field value falls outside three standard deviations from the training mean, THE System SHALL clip the value to the three-standard-deviation boundary.
3. THE System SHALL encode the target label (diabetes type) as an integer in the range [0, 3] using a consistent, reproducible label mapping.
4. WHEN the training dataset contains class imbalance where any class represents fewer than 15% of samples, THE System SHALL apply SMOTE to the training split to produce a balanced training set.
5. WHERE class weighting is configured, THE System SHALL pass inverse-frequency class weights to the Classifier during training.

---

### Requirement 3: NLP Preprocessing

**User Story:** As a data scientist, I want clinical notes cleaned and segmented, so that NLP models receive well-formed text inputs.

#### Acceptance Criteria

1. THE NLP_Pipeline SHALL convert all Clinical_Notes text to lowercase before further processing.
2. THE NLP_Pipeline SHALL replace all PHI placeholder tokens (e.g., `[NAME]`, `[DOB]`, `[ID]`) with a generic `<PHI>` token.
3. THE NLP_Pipeline SHALL segment each Clinical_Notes string into sentences using spaCy's sentence boundary detection.
4. WHEN a Clinical_Notes string is empty or null, THE NLP_Pipeline SHALL substitute a configurable default string (e.g., `"no clinical notes provided"`) and log a warning.
5. THE NLP_Pipeline SHALL extract named entities (symptoms, medications, lab values) from Clinical_Notes using a spaCy NER model and store them as structured metadata alongside each record.

---

### Requirement 4: Feature Extraction and Embedding

**User Story:** As a data scientist, I want structured features and text embeddings combined into a single feature vector, so that models can learn from both data modalities.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL produce a TF-IDF baseline embedding of dimension ≤ 10,000 from preprocessed Clinical_Notes.
2. THE Feature_Extractor SHALL produce a ClinicalBERT embedding by mean-pooling the last hidden layer of a fine-tuned `emilyalsentzer/Bio_ClinicalBERT` model, yielding a 768-dimensional vector per record.
3. THE Feature_Extractor SHALL concatenate the normalized structured feature vector with the selected text embedding (TF-IDF or BERT) to produce a single combined feature vector.
4. WHEN the embedding mode is set to `"tfidf"`, THE Feature_Extractor SHALL use TF-IDF embeddings; WHEN set to `"bert"`, THE Feature_Extractor SHALL use ClinicalBERT embeddings.
5. THE Feature_Extractor SHALL serialize the fitted TF-IDF vectorizer and BERT tokenizer configuration to disk so that inference uses identical transformations to training.

---

### Requirement 5: Model Training

**User Story:** As a data scientist, I want to train baseline and advanced models with experiment tracking, so that I can compare approaches and reproduce results.

#### Acceptance Criteria

1. THE System SHALL train a Logistic Regression baseline and a Decision Tree baseline using scikit-learn on the combined feature vectors.
2. THE System SHALL fine-tune a BERT_Classifier on Clinical_Notes using the HuggingFace Transformers library for multi-class classification across four diabetes types.
3. THE System SHALL train an Ensemble model that combines probability outputs from the structured Classifier and the BERT_Classifier using a configurable weighting scheme.
4. WHEN a training run completes, THE Experiment_Tracker SHALL log all hyperparameters, training metrics, and model artifacts to an MLflow experiment named `"diabetes-classifier"`.
5. THE System SHALL version all training datasets and model artifacts using DVC, storing metadata in `.dvc` files committed to the repository.
6. WHEN training is invoked with a fixed random seed, THE System SHALL produce identical model weights and evaluation metrics across repeated runs on the same dataset.

---

### Requirement 6: Model Evaluation

**User Story:** As a clinical data scientist, I want rigorous, medical-grade evaluation of all models, so that I can assess safety and performance before deployment.

#### Acceptance Criteria

1. THE System SHALL evaluate all models using stratified 5-fold cross-validation, ensuring each fold preserves the class distribution of the full dataset.
2. THE System SHALL compute and report macro F1-score, weighted F1-score, multi-class ROC-AUC (one-vs-rest), per-class precision-recall curves, a confusion matrix, and a calibration curve for each model.
3. THE System SHALL report per-class recall separately for Type 1 and Gestational diabetes classes in all evaluation outputs.
4. WHEN any model's macro F1-score on the validation set falls below 0.70, THE System SHALL log a warning indicating the model does not meet the minimum performance threshold.
5. THE System SHALL save all evaluation plots (confusion matrix, ROC curves, calibration curve) as PNG files to the `experiments/` directory, named with the model name and timestamp.

---

### Requirement 7: Model Interpretability

**User Story:** As a clinician, I want to understand why the model made a specific prediction, so that I can trust and verify the output.

#### Acceptance Criteria

1. WHEN a prediction is made by the structured Classifier, THE System SHALL compute SHAP values for that prediction and return the top 5 most influential features with their SHAP scores.
2. WHEN a prediction is made by a Decision Tree Classifier, THE System SHALL return the decision path as an ordered list of feature-threshold conditions leading to the predicted class.
3. WHEN a prediction is made by the BERT_Classifier, THE System SHALL extract the top 5 attention-weighted tokens from the final attention layer and include them in the explanation output.
4. THE System SHALL include the interpretability output in the API response under an `"explanation"` key as a structured JSON object.

---

### Requirement 8: Inference API

**User Story:** As a backend engineer, I want a REST API that accepts patient data and returns a diabetes classification with confidence and explanation, so that downstream applications can integrate predictions.

#### Acceptance Criteria

1. THE Inference_API SHALL expose a `POST /predict` endpoint that accepts a JSON body containing at minimum `age` (integer), `bmi` (float), and `notes` (string).
2. WHEN a valid request is received at `POST /predict`, THE Inference_API SHALL return a JSON response containing `prediction` (string label), `confidence` (float in [0.0, 1.0]), and `explanation` (object) within 2000 milliseconds under normal load.
3. WHEN a request body is missing a required field, THE Inference_API SHALL return HTTP 422 with a JSON error body listing each missing field.
4. WHEN an internal error occurs during inference, THE Inference_API SHALL return HTTP 500 with a JSON error body and log the full stack trace.
5. THE Inference_API SHALL expose a `GET /health` endpoint that returns HTTP 200 and `{"status": "ok"}` when the service is running and the model is loaded.
6. THE Inference_API SHALL log each request's input fields (excluding raw notes content), predicted label, confidence score, and latency in milliseconds to a structured JSON log.

---

### Requirement 9: Data Serialization Round-Trip

**User Story:** As a data engineer, I want serialized feature vectors and model artifacts to be deserializable to their original form, so that inference is consistent with training.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL serialize fitted preprocessing objects (scaler, TF-IDF vectorizer, label encoder) to disk in a format that, when deserialized, produces byte-identical transformation outputs on the same input data.
2. FOR ALL valid structured feature records, serializing then deserializing the preprocessing pipeline SHALL produce feature vectors equal to those produced without serialization (round-trip property).
3. FOR ALL valid Clinical_Notes strings, encoding with the BERT tokenizer then decoding SHALL recover the original token sequence (round-trip property).
4. WHEN a serialized model artifact file is corrupted or missing, THE System SHALL raise a descriptive error identifying the missing artifact path before attempting inference.

---

### Requirement 10: Deployment and Containerization

**User Story:** As a DevOps engineer, I want the system packaged as a Docker container, so that it can be deployed consistently across environments.

#### Acceptance Criteria

1. THE System SHALL include a `Dockerfile` that builds a runnable image containing all Python dependencies, model artifacts, and the Inference_API.
2. WHEN the Docker container starts, THE Inference_API SHALL be available on a configurable port (default 8000) within 60 seconds.
3. THE System SHALL include a `requirements.txt` listing all Python dependencies with pinned versions.
4. WHEN the `GET /health` endpoint is called after container startup, THE Inference_API SHALL return HTTP 200, confirming the container is healthy.

---

### Requirement 11: Testing

**User Story:** As a software engineer, I want automated unit and integration tests, so that regressions are caught before deployment.

#### Acceptance Criteria

1. THE System SHALL include unit tests for all preprocessing functions, covering at least: missing value handling, PHI replacement, normalization, and label encoding.
2. THE System SHALL include unit tests for model inference that verify the output schema (prediction string, confidence float, explanation object) on synthetic inputs.
3. THE System SHALL include integration tests for the `POST /predict` and `GET /health` endpoints using pytest and the FastAPI `TestClient`.
4. WHEN all tests are run with `pytest tests/`, THE System SHALL report zero failures on a clean environment with all dependencies installed.

---

### Requirement 12: Experiment Tracking and Model Versioning

**User Story:** As a data scientist, I want all experiments logged and datasets versioned, so that any result is reproducible.

#### Acceptance Criteria

1. WHEN a training run starts, THE Experiment_Tracker SHALL create a new MLflow run under the `"diabetes-classifier"` experiment and log the run ID to stdout.
2. THE Experiment_Tracker SHALL log at minimum: model type, embedding mode, random seed, number of training samples, and all hyperparameters as MLflow parameters.
3. THE Experiment_Tracker SHALL log at minimum: macro F1, weighted F1, ROC-AUC, and per-class recall as MLflow metrics at the end of each fold.
4. THE System SHALL register the best-performing model (highest macro F1 on validation) in the MLflow Model Registry under the name `"diabetes-classifier-production"`.

---

### Requirement 13: Data Drift Detection (Bonus)

**User Story:** As an MLOps engineer, I want the system to detect when incoming data distributions shift from training distributions, so that model degradation is caught early.

#### Acceptance Criteria

1. THE Drift_Detector SHALL compute the Population Stability Index (PSI) for each structured numeric feature between the training distribution and a provided batch of inference requests.
2. WHEN the PSI for any feature exceeds 0.2, THE Drift_Detector SHALL log a warning identifying the feature name and its PSI value.
3. THE Inference_API SHALL expose a `POST /drift-report` endpoint that accepts a batch of records and returns a JSON object with per-feature PSI scores.

---

### Requirement 14: Role-Based Access Control (Bonus)

**User Story:** As a security engineer, I want API endpoints protected by role-based access, so that only authorized users can access sensitive prediction and admin functions.

#### Acceptance Criteria

1. THE Inference_API SHALL require a valid JWT bearer token on all endpoints except `GET /health`.
2. WHEN a request is received without a valid JWT token, THE Inference_API SHALL return HTTP 401.
3. WHERE the `"admin"` role is present in the JWT claims, THE Inference_API SHALL permit access to model retraining and drift-report endpoints.
4. WHERE only the `"doctor"` role is present in the JWT claims, THE Inference_API SHALL permit access to `POST /predict` and deny access to retraining and drift-report endpoints, returning HTTP 403.
