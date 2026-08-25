# Student Performance Prediction

A deep neural network built with TensorFlow/Keras that classifies students into one of five grade classes (**A, B, C, D, F**) based on academic and behavioral features. The pipeline covers preprocessing, model training with regularization, evaluation, and permutation-based feature importance analysis.

---

## Overview

Given a student's demographic, academic, and behavioral attributes, the model predicts their expected grade class. The project is structured as an end-to-end pipeline: data loading → preprocessing → model training → evaluation → feature importance → artifact export for downstream inference.

## Dataset

- **Source:** [Students Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset) (Kaggle, by Rabie El Kharoua)
- **Target:** `GradeClass` — 5 classes:
  | Class | Meaning       |
  | ----- | ------------- |
  | 0     | A (Excellent) |
  | 1     | B (Good)      |
  | 2     | C (Average)   |
  | 3     | D (Below Avg) |
  | 4     | F (At-Risk)   |
- **Features:** Study hours, absences, tutoring, parental support/education, extracurriculars, and other academic/behavioral signals. `StudentID` is dropped before training.

## Model Architecture

A fully connected feed-forward network optimized for tabular multi-class classification:

```
Input (n_features)
    │
Dense(64, ReLU) → BatchNorm → Dropout(0.3)
    │
Dense(32, ReLU) → BatchNorm → Dropout(0.2)
    │
Dense(16, ReLU)
    │
Dense(5, Softmax)
```

- **Optimizer:** Adam (`lr = 0.001`)
- **Loss:** Categorical cross-entropy
- **Metric:** Accuracy
- **Regularization:** Dropout, BatchNorm, and early stopping on validation loss

## Pipeline

1. **Preprocessing**
   - Drop `StudentID`
   - Label-encode categorical columns
   - Standardize features with `StandardScaler`
   - One-hot encode the target with `to_categorical`
2. **Splits**
   - Train: 80% (stratified on target)
   - Validation: 10%
   - Test: 10%
3. **Training**
   - Up to 200 epochs, batch size 64
   - `EarlyStopping` (patience = 15, restores best weights)
   - `ModelCheckpoint` saves the best model by validation accuracy
4. **Evaluation**
   - Test accuracy and loss
   - Per-class precision/recall/F1 via `classification_report`
   - Confusion matrix heatmap
5. **Feature Importance**
   - Permutation-based importance on the held-out test set
   - Top-10 features exported to CSV and visualized

## Repository Structure

```
.
├── student-performance-notebook.ipynb   # Main training + evaluation notebook
├── artifacts/
│   ├── student_model.h5                 # Trained Keras model
│   ├── best_model.h5                    # Best checkpoint by val accuracy
│   ├── scaler.pkl                       # Fitted StandardScaler
│   ├── label_encoders.pkl               # Fitted LabelEncoders (per categorical col)
│   ├── feature_cols.pkl                 # Feature column order (for inference)
│   └── feature_importance.csv           # Permutation importance scores
├── figures/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── README.md
```

## Getting Started

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

Tested with Python 3.10+ and TensorFlow 2.x.

### Run the Notebook

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).
2. Update the CSV path in the notebook if you're running locally (the notebook currently expects the Kaggle input path).
3. Open `student-performance-notebook.ipynb` and run all cells.

The notebook will train the model, print evaluation metrics, save the figures, and export all inference artifacts.

### Inference (example)

```python
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load artifacts
model = load_model('artifacts/student_model.h5')
with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('artifacts/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('artifacts/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Prepare a new sample (as a DataFrame with the same columns as training)
sample = pd.DataFrame([{
    # 'ColumnName': value,
    # ...
}])

# Apply the same label encoders to categorical columns
for col, le in label_encoders.items():
    if col in sample.columns:
        sample[col] = le.transform(sample[col].astype(str))

X = scaler.transform(sample[feature_cols].values)
probs = model.predict(X)
pred_class = int(np.argmax(probs, axis=1)[0])

grade_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
print(f"Predicted grade: {grade_labels[pred_class]}  (confidence: {probs[0][pred_class]:.2%})")
```

## Results

Training curves, confusion matrix, and top feature importances are saved to the `figures/` directory after running the notebook. The permutation-importance ranking highlights which academic and behavioral factors most strongly drive the model's predictions.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data / ML:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn

## Future Work

- Compare against gradient-boosted baselines (XGBoost, LightGBM) on the same splits
- Add class-weighted loss or SMOTE to address any grade-class imbalance
- Wrap the inference code as a FastAPI service for real-time predictions
- Hyperparameter search via Keras Tuner or Optuna

## License

MIT — feel free to use, modify, and share with attribution.
