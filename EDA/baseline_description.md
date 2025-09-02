## 🧪 Baseline CNN for Emotion Recognition

### 1. Data Preparation
- **Dataset:** [Natural Human Face Images for Emotion Recognition](https://www.kaggle.com/datasets) (Kaggle).
- **Splits:**
  - 70% training
  - 15% validation
  - 15% test
- **Preprocessing:**
  - Resized all images to **224×224 RGB**.
  - Normalized pixel values to `[0, 1]` using `Rescaling(1./255)`.
  - Optimized pipeline with `.cache()`, `.shuffle()`, and `.prefetch()` for performance.

### 2. Model Architecture
Custom CNN built with `tf.keras.Sequential`:


Input: (224, 224, 3)
↓
Conv2D(16, kernel=4, relu, padding='same')
BatchNormalization
MaxPooling2D
↓
Conv2D(32, kernel=3, relu, padding='same')
BatchNormalization
MaxPooling2D
↓
Conv2D(64, kernel=3, relu, padding='same')
BatchNormalization
MaxPooling2D
↓
GlobalAveragePooling2D
Dense(64, relu)
Dropout(0.25)
Dense(8, softmax) <-- classifier for 8 emotions


- **Optimizer:** Adam (`lr=1e-3`)
- **Loss:** Categorical Crossentropy
- **Metric:** Accuracy

### 3. Training Setup
- **Epochs:** up to 50
- **Callbacks:**
  - `EarlyStopping` (patience=10, restore best weights)
  - `ModelCheckpoint` (saves best model by val accuracy)

### 4. Results (Baseline)
- **Train accuracy:** ~29%
- **Validation accuracy:** ~29%
- **Test accuracy:** ~25%
- **Loss:** ~1.86–1.94

**Interpretation:**
- Performance is above random guessing (~12.5% for 8 classes).
- The model shows no strong overfitting but struggles to capture complex emotional features.
- Indicates the baseline CNN is **too simple** → good as a performance floor.

### 5. Next Steps
- Transition to **transfer learning** with pretrained backbones (MobileNetV2, EfficientNet, ResNet).
- Add **data augmentation** and experiment with **class balancing**.
- Use this baseline as a benchmark for measuring improvements with advanced models.
