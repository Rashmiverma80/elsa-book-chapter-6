"""
AI-based Structural Health Monitoring of Digital Concrete
CNN-based Crack Detection using SDNET2018 Dataset

This script performs:
1. Dataset loading and preprocessing
2. CNN model development
3. Model training and validation
4. Performance evaluation using Confusion Matrix and ROC Curve

Corresponding Chapter Sections:
- Data Collection
- Data Processing
- Model Development
- Evaluation Metrics
"""

# --------------------------------------------------------
# SECTION 1: Import Required Libraries
# --------------------------------------------------------

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

# --------------------------------------------------------
# SECTION 2: Dataset Loading (SDNET2018)
# --------------------------------------------------------
# Expected folder structure:
# SDNET2018/
# ├── Crack/
# └── NonCrack/

DATASET_PATH = "SDNET2018"
IMG_SIZE = 128

images = []
labels = []

for label, category in enumerate(["NonCrack", "Crack"]):
    category_path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

# --------------------------------------------------------
# SECTION 3: Data Preprocessing
# --------------------------------------------------------
# Normalization and reshaping for CNN input

X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(labels, num_classes=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------------
# SECTION 4: CNN Model Development
# --------------------------------------------------------
# Model architecture for crack detection

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------------
# SECTION 5: Model Training
# --------------------------------------------------------
# Training with validation split

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --------------------------------------------------------
# SECTION 6: Model Prediction
# --------------------------------------------------------
# Generate predictions for evaluation

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# --------------------------------------------------------
# SECTION 7: Evaluation – Confusion Matrix
# --------------------------------------------------------
# Visualizing classification performance

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Crack", "Crack"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix for CNN-based Crack Detection")
plt.show()

# --------------------------------------------------------
# SECTION 8: Evaluation – ROC Curve
# --------------------------------------------------------
# Computing ROC-AUC score

fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="CNN (AUC = %0.2f)" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Concrete Crack Detection")
plt.legend(loc="lower right")
plt.show()
