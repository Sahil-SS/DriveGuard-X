# DriveGuard-X IMU AI

## Driving Behavior Classification using Machine Learning

**Project:** DriveGuard-X  
**Module:** IMU AI  
**Primary Task:** Driving Behavior Classification  
**Dataset:** Public Driving Behavior Dataset based on 6-axis IMU measurements  
**Base Research Paper:** Yüksel and Atmaca, *Driver's black box: a system for driver risk assessment using machine learning and fuzzy logic*  
**Final Model:** XGBoost Classifier  
**Implementation Environment:** Python / Google Colab Notebook  
**Intended Integration:** FastAPI + MATLAB/Simulink + ESP32-based DriveGuard-X prototype

---

# 1. Overview

The IMU AI module is one of the core intelligence components of **DriveGuard-X**, a context-aware adaptive battery protection system designed for an electric-vehicle-like platform.

The purpose of this module is to estimate the **driving behavior of the vehicle** from inertial measurements collected using a 6-axis IMU.

The model uses:
- 3-axis acceleration
- 3-axis angular velocity

to identify the driving-behavior class associated with the observed vehicle motion.

The estimated driving context is subsequently intended to be combined with:
- Battery condition
- Road condition
- Model confidence
- Temporal information

inside the higher-level **multimodal fusion and causal attribution system**.

The overall concept is:

```text
                    IMU Measurements
                           │
                           ▼
                 ┌─────────────────────┐
                 │   Sliding Window    │
                 │                     │
                 │  AccX/Y/Z           │
                 │  GyroX/Y/Z          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Extraction  │
                 │                     │
                 │ Mean                │
                 │ Skewness            │
                 │ Kurtosis            │
                 │ Min / Max           │
                 │ Median              │
                 │ Standard Deviation  │
                 └──────────┬──────────┘
                            │
                            ▼
                    42 IMU Features
                            │
                            ▼
                 ┌─────────────────────┐
                 │      IMU AI         │
                 │  XGBoost Classifier │
                 └──────────┬──────────┘
                            │
                            ▼
                  Driving Behavior Class
                            │
                            ▼
                 ┌─────────────────────┐
                 │ DriveGuard-X Fusion │
                 │ & Attribution Layer │
                 └──────────┬──────────┘
                            │
                            ▼
                  Adaptive Protection
```

The IMU AI model is **not intended to independently make safety-critical decisions**. It provides driving-context information to the larger DriveGuard-X protection architecture.

---

# 2. Project Motivation

A conventional battery protection system primarily observes battery-side parameters such as:

- Voltage
- Current
- Temperature
- SOC
- SOH

and generally reacts when a predefined threshold is exceeded.

However, a high-current or high-stress battery event does not necessarily have a single cause.

For example, increased battery stress may occur because of:
- Aggressive driving
- Sudden acceleration
- Sudden deceleration
- Sharp vehicle movement
- Rough road conditions
- An already degraded battery

Therefore, observing only battery-side information may not be sufficient to understand **why** the battery is experiencing stress.

DriveGuard-X introduces vehicle-motion information as an additional context source.

The intended system is not simply:

```text
Battery Stress
      ↓
Threshold Exceeded
      ↓
Protection
```

Instead, it aims toward:

```text
Battery condition
        +
Driver behavior
        +
Road condition
        ↓
Causal interpretation
        ↓
Adaptive protection
```

The IMU AI module is responsible for the **driving-behavior component** of this architecture.

---

# 3. Research Baseline

The IMU AI development was based on:

> **Asim Sinan Yüksel and Sedat Atmaca**,  
> *Driver's black box: a system for driver risk assessment using machine learning and fuzzy logic*,  
> Journal of Intelligent Transportation Systems, 2021, Vol. 25, No. 5, pp. 482–500.

**DOI:** `10.1080/15472450.2020.1852083`

The paper uses inertial measurements to identify driving behaviors and evaluates multiple machine-learning classification algorithms.

The methodology is based on:

```text
6-axis IMU
   │
   ├── Accelerometer
   │   ├── AccX
   │   ├── AccY
   │   └── AccZ
   │
   └── Gyroscope
       ├── GyroX
       ├── GyroY
       └── GyroZ
```

The DriveGuard-X model follows this general feature-based approach while developing and validating a deployment-oriented IMU model.

---

# 4. Development Philosophy

The objective was **not simply to reproduce the reference paper's reported results**.

The development strategy was:

```text
Base Research Paper
        │
        ▼
Dataset Inspection
        │
        ▼
Raw IMU Analysis
        │
        ▼
Feature Verification
        │
        ▼
Feature Engineering
        │
        ▼
Leakage-Aware Evaluation
        │
        ▼
Baseline ML Models
        │
        ▼
XGBoost Optimization
        │
        ▼
Feature Reduction
        │
        ▼
Temporal Distribution Study
        │
        ▼
Final Model Selection
        │
        ▼
Untouched Test Set
        │
        ▼
Model Export
        │
        ▼
Raw IMU Pipeline Validation
        │
        ▼
Final IMU AI Model
```

This development process was designed to answer two questions:

### Question 1
Can an IMU-based machine-learning model reliably classify driving behavior from accelerometer and gyroscope measurements?

### Question 2
Does the model maintain performance when evaluated using a realistic temporal split rather than a simple random train/test split?

The second question is important because adjacent sliding-window samples can contain highly overlapping sensor information.

---

# 5. Dataset

The IMU model uses a public driving-behavior dataset containing 6-axis inertial measurements.

The raw dataset contains:

```text
1114 samples
7 columns
```

The raw columns are:

```text
Target(Class)
GyroX
GyroY
GyroZ
AccX
AccY
AccZ
```

The six sensor channels are:

### Accelerometer
```text
AccX
AccY
AccZ
```

### Gyroscope
```text
GyroX
GyroY
GyroZ
```

The target variable contains four numerical classes:

```text
1
2
3
4
```

The numerical labels are retained as provided by the dataset. The exact semantic mapping of the numerical labels should be verified from the source dataset before being hard-coded into a deployment system.

---

# 6. Dataset Structure

The raw dataset consists of sequential IMU measurements.

A representative record contains:

```text
Target(Class)
GyroX
GyroY
GyroZ
AccX
AccY
AccZ
```

The raw measurements are converted into statistical features using a short sliding window.

```text
Raw IMU Dataset
       │
       ▼
6-axis Sensor Measurements
       │
       ▼
4-Sample Sliding Window
       │
       ▼
Statistical Feature Extraction
       │
       ▼
60 Original Features
       │
       ▼
Feature Reduction
       │
       ▼
42 Final Features
       │
       ▼
XGBoost Classifier
       │
       ▼
Driving Behavior Class
```

---

# 7. Raw IMU Data Quality

The raw IMU dataset was systematically inspected before model development.

```text
Samples              : 1114
Columns              : 7
Missing Values       : None
Duplicate Rows       : None
Infinite Values      : None
Target Classes       : 1, 2, 3, 4
```

This established that the raw dataset could be used for feature construction without missing-value or duplicate-row treatment.

---

# 8. Sliding-Window Feature Generation

The feature dataset contains:

```text
1102 samples
61 columns
```

consisting of:

```text
1 Target column
+
60 statistical features
```

The feature-generation process was mathematically verified against the raw sensor data.

The verification established that the feature rows correspond to:

```text
4-sample sliding window
Stride = 1
```

generated independently within each class.

Because a 4-sample window is used, each class loses three samples during feature generation:

```text
Raw samples       : 1114
Feature samples   : 1102
```

The file name `features_14.csv` was **not interpreted as a 14-second deployment window**. Mathematical reconstruction showed that the actual feature-generation process corresponds to a 4-sample sliding window.

---

# 9. Class Distribution

The final feature dataset contains:

| Class | Samples |
|---|---:|
| Class 1 | 249 |
| Class 2 | 285 |
| Class 3 | 347 |
| Class 4 | 221 |
| **Total** | **1102** |

The class distribution is therefore not perfectly balanced.

---

# 10. Feature Engineering

The original feature dataset contains **60 statistical features**.

The features are calculated independently for:

```text
Accelerometer:
AccX
AccY
AccZ

Gyroscope:
GyroX
GyroY
GyroZ
```

The statistical feature groups include:

```text
Mean
Covariance
Skewness
Kurtosis
Sum
Minimum
Maximum
Variance
Median
Standard Deviation
```

These features capture different aspects of vehicle motion within each measurement window.

---

# 11. Accelerometer Features

The original accelerometer feature groups include:

```text
AccMeanX
AccMeanY
AccMeanZ

AccCovX
AccCovY
AccCovZ

AccSkewX
AccSkewY
AccSkewZ

AccKurtX
AccKurtY
AccKurtZ

AccSumX
AccSumY
AccSumZ

AccMinX
AccMinY
AccMinZ

AccMaxX
AccMaxY
AccMaxZ

AccVarX
AccVarY
AccVarZ

AccMedianX
AccMedianY
AccMedianZ

AccStdX
AccStdY
AccStdZ
```

---

# 12. Gyroscope Features

The original gyroscope feature groups include:

```text
GyroMeanX
GyroMeanY
GyroMeanZ

GyroCovX
GyroCovY
GyroCovZ

GyroSkewX
GyroSkewY
GyroSkewZ

GyroSumX
GyroSumY
GyroSumZ

GyroKurtX
GyroKurtY
GyroKurtZ

GyroMinX
GyroMinY
GyroMinZ

GyroMaxX
GyroMaxY
GyroMaxZ

GyroVarX
GyroVarY
GyroVarZ

GyroMedianX
GyroMedianY
GyroMedianZ

GyroStdX
GyroStdY
GyroStdZ
```

---

# 13. Feature Correlation Analysis

Correlation analysis was performed to identify highly redundant features.

Using:

```text
|correlation| >= 0.95
```

the analysis identified:

```text
22 highly correlated feature pairs
```

Strong relationships were observed between statistical features such as:

```text
Covariance ↔ Variance
Mean ↔ Sum
Variance ↔ Standard Deviation
```

This indicated substantial redundancy in the original 60-dimensional representation.

---

# 14. Final Feature Set

Three feature configurations were evaluated:

```text
60 features
    ↓
44 features
    ↓
42 features
```

The final 42-feature model retains:

```text
Mean
Skewness
Kurtosis
Minimum
Maximum
Median
Standard Deviation
```

for all six IMU axes.

The following groups were removed:

```text
Covariance
Sum
Variance
```

The reduction produced a more compact representation and improved performance on the final temporal test set.

---

# 15. Complete 42-Feature Set

The final feature order is fixed and must be preserved during deployment.

```text
1.  AccMeanX
2.  AccMeanY
3.  AccMeanZ
4.  AccSkewX
5.  AccSkewY
6.  AccSkewZ
7.  AccKurtX
8.  AccKurtY
9.  AccKurtZ
10. AccMinX
11. AccMinY
12. AccMinZ
13. AccMaxX
14. AccMaxY
15. AccMaxZ
16. AccMedianX
17. AccMedianY
18. AccMedianZ
19. AccStdX
20. AccStdY
21. AccStdZ
22. GyroMeanX
23. GyroMeanY
24. GyroMeanZ
25. GyroSkewX
26. GyroSkewY
27. GyroSkewZ
28. GyroKurtX
29. GyroKurtY
30. GyroKurtZ
31. GyroMinX
32. GyroMinY
33. GyroMinZ
34. GyroMaxX
35. GyroMaxY
36. GyroMaxZ
37. GyroMedianX
38. GyroMedianY
39. GyroMedianZ
40. GyroStdX
41. GyroStdY
42. GyroStdZ
```

The exact feature order is stored in:

```text
imu_feature_list.json
```

---

# 16. Data Splitting Strategy

A simple random train/test split can produce overly optimistic results for sliding-window time-series data.

For example:

```text
Window 1 ───── Training
Window 2 ───── Training
Window 3 ───── Testing
```

may allow highly similar or overlapping measurements to appear in both training and testing.

Therefore, DriveGuard-X uses a **temporal, leakage-aware evaluation strategy**.

The final split contains:

```text
Training samples : 880
Testing samples  : 210
```

A purge gap was applied between training and testing portions of each class because each feature vector is generated from four consecutive raw samples.

The final test set remained untouched during hyperparameter optimization.

---

# 17. Final Temporal Split

The final split was performed independently within each class.

```text
Class 1:
Training : 199
Testing  : 47

Class 2:
Training : 228
Testing  : 54

Class 3:
Training : 277
Testing  : 67

Class 4:
Training : 176
Testing  : 42
```

Total:

```text
Training : 880
Testing  : 210
```

---

# 18. Baseline Model Experiment

The following classifiers were evaluated:

```text
Random Forest
XGBoost
Support Vector Machine
Logistic Regression
```

The objective was to determine the strongest baseline before optimization.

---

# 19. Baseline Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 70.00% | 82.33% | 75.54% | 68.96% | 65.32% |
| SVM | 68.10% | 71.36% | 73.55% | 67.77% | 63.96% |
| Logistic Regression | 67.14% | 77.55% | 72.62% | 66.37% | 62.64% |
| **XGBoost** | **71.43%** | **80.63%** | **76.50%** | **71.00%** | **67.95%** |

XGBoost produced the strongest baseline accuracy and macro F1 and was selected for further optimization.

---

# 20. XGBoost Optimization

Hyperparameter optimization was performed using a temporal validation split inside the training dataset.

```text
Training:
659 samples

Validation:
209 samples

Final Test:
210 samples
```

A total of:

```text
54 configurations
```

were evaluated.

The best validation configuration achieved:

```text
Validation Accuracy        : 91.87%
Validation Macro F1        : 90.51%
Validation Macro Precision : 93.12%
Validation Macro Recall    : 90.07%
Validation Weighted F1     : 91.57%
```

The final test set was not used during optimization.

---

# 21. Final XGBoost Hyperparameters

```text
n_estimators      = 300
max_depth         = 3
learning_rate     = 0.03
subsample         = 0.8
colsample_bytree  = 0.8
min_child_weight  = 1
```

---

# 22. Feature Reduction Experiments

Three configurations were compared:

### 60-feature model
Original statistical feature representation.

### 44-feature model
Training-only correlation-based reduction using:

```text
|correlation| >= 0.95
```

### 42-feature model
Retained:

```text
Mean
Skewness
Kurtosis
Minimum
Maximum
Median
Standard Deviation
```

for all six axes.

---

# 23. Feature Reduction Results

| Feature Configuration | Features | Accuracy | Macro F1 | Weighted F1 | Class 3 F1 |
|---|---:|---:|---:|---:|---:|
| Original | 60 | 72.86% | 72.71% | 69.83% | 38.10% |
| Correlation Reduced | 44 | 73.81% | 73.87% | 71.23% | 41.86% |
| **Final Reduced** | **42** | **74.29%** | **74.62%** | **72.24%** | **45.45%** |

Compared with the 60-feature model, the final 42-feature model achieved:

```text
Accuracy improvement       : +1.43 percentage points
Macro F1 improvement       : +1.91 percentage points
Weighted F1 improvement    : +2.41 percentage points
Class 3 recall improvement : +5.97 percentage points
Class 3 F1 improvement     : +7.35 percentage points
```

The 42-feature representation was therefore selected as the final configuration.

---

# 24. Temporal Distribution Analysis

Temporal analysis revealed significant distribution changes in several features, particularly for Class 3.

Examples include:

| Feature | Early → Late Change |
|---|---:|
| GyroMaxZ | approximately −47% |
| AccStdY | approximately −38.5% |
| GyroStdZ | approximately −43.7% |
| AccVarY | approximately −47% |
| GyroMeanZ | approximately −36.3% |

These changes indicate that the sensor characteristics are not completely stationary throughout the dataset.

The late portion of Class 3 partially moves toward a lower-dynamic regime, increasing overlap with Class 1.

This temporal distribution shift helps explain the difficulty of classification under leakage-aware temporal evaluation.

---

# 25. Final Model Architecture

```text
                       Raw IMU Data
                            │
                            ▼
                 ┌──────────────────────┐
                 │ 4-Sample Window      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Extraction   │
                 │                      │
                 │ Accelerometer        │
                 │ Gyroscope            │
                 │                      │
                 │ Mean                 │
                 │ Skewness             │
                 │ Kurtosis             │
                 │ Min / Max            │
                 │ Median               │
                 │ Standard Deviation   │
                 └──────────┬───────────┘
                            │
                            ▼
                     42 Features
                            │
                            ▼
                 ┌──────────────────────┐
                 │      XGBoost         │
                 │     Classifier       │
                 └──────────┬───────────┘
                            │
                            ▼
                  Driving Behavior Class
                            │
                            ▼
                 DriveGuard-X Fusion Layer
```

---

# 26. Final Training and Testing Strategy

### Training Set

```text
880 samples
42 features
```

The training data was used for:

- Model training
- Internal temporal validation
- Hyperparameter optimization
- Feature-selection experiments
- Model comparison

### Untouched Test Set

```text
210 samples
42 features
```

The test set was reserved for final evaluation and was not used during hyperparameter optimization.

---

# 27. Final Model Performance

The final 42-feature XGBoost model achieved:

| Metric | Result |
|---|---:|
| **Accuracy** | **74.29%** |
| **Macro Precision** | **82.18%** |
| **Macro Recall** | **78.74%** |
| **Macro F1** | **74.62%** |
| **Weighted F1** | **72.24%** |

These values are obtained from the untouched temporal test set.

---

# 28. Final Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Class 1 | 45.98% | 85.11% | 59.70% | 47 |
| Class 2 | 100.00% | 100.00% | 100.00% | 54 |
| Class 3 | 95.24% | 29.85% | 45.45% | 67 |
| Class 4 | 87.50% | 100.00% | 93.33% | 42 |

The primary weakness is Class 3 recall.

---

# 29. Final Confusion Matrix

```text
                 Predicted
              Class 1  Class 2  Class 3  Class 4

Actual Class 1    40        0        1        6
Actual Class 2     0       54        0        0
Actual Class 3    47        0       20        0
Actual Class 4     0        0        0       42
```

The dominant error is:

```text
Class 3 → Class 1
```

with:

```text
47 of 67 Class 3 samples
```

being classified as Class 1.

This is the primary area for future improvement.

---

# 30. Final XGBoost Feature Importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | GyroMaxZ | 0.122047 |
| 2 | AccStdX | 0.111149 |
| 3 | AccMaxY | 0.108710 |
| 4 | GyroMinZ | 0.101240 |
| 5 | AccMedianX | 0.099513 |
| 6 | AccStdY | 0.094560 |
| 7 | GyroMeanZ | 0.088998 |
| 8 | AccMeanY | 0.031576 |
| 9 | AccMinY | 0.030052 |
| 10 | AccMaxX | 0.029546 |
| 11 | AccMedianY | 0.025969 |
| 12 | GyroStdZ | 0.023849 |
| 13 | GyroMinY | 0.021983 |
| 14 | AccMeanX | 0.020877 |
| 15 | GyroKurtZ | 0.015338 |

Feature importance describes **model behavior**, not direct physical causality.

---

# 31. Raw IMU to Feature Pipeline Validation

A critical validation was performed to ensure that raw IMU measurements can be converted into exactly the same feature representation used during training.

```text
Raw sensor data
       │
       ▼
4 consecutive samples
       │
       ▼
42-feature reconstruction
       │
       ▼
Comparison with original feature row
```

Validation results:

```text
Features matched : 42 / 42

Maximum absolute difference:
0.000000000047

Mean absolute difference:
0.000000000001
```

Therefore:

```text
Raw IMU
   ↓
4-sample window
   ↓
42-feature extraction
   ↓
Final feature ordering
```

has been numerically validated for deployment.

---

# 32. Model Export and Verification

The final model was exported with:

```text
driveguard_xgb_final.json
driveguard_xgb_final.joblib
imu_feature_list.json
model_metadata.json
```

The exported model was subsequently reloaded and tested.

The verification confirmed:

```text
Original model predictions
          =
Reloaded model predictions
```

Therefore, the saved model artifact can reproduce the predictions of the trained model.

---

# 33. Deployment Considerations

The IMU model is structured around a short sliding window and therefore has a clear path toward real-time software integration.

The intended deployment pipeline is:

```text
ESP32
 │
 ▼
MPU6050
 │
 ▼
Continuous IMU Sampling
 │
 ▼
4-Sample Sliding Window
 │
 ▼
42-Feature Extraction
 │
 ▼
XGBoost Model
 │
 ▼
Driving Behavior Class
 │
 ▼
FastAPI / Fusion Layer
```

Actual deployment will require validation of:

- Sampling rate
- Sensor orientation
- Sensor calibration
- Window timing
- Computational latency
- Communication latency
- Model inference time

---

# 34. Integration with DriveGuard-X

The IMU AI model is not intended to operate independently.

```text
                         DRIVEGUARD-X
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   Battery Sensors       MPU6050 IMU        Microphone +
          │                   │              Road Sensors
          ▼                   ▼                   ▼
     Battery AI          Driving AI           Road AI
          │                   │                   │
          ▼                   ▼                   ▼
        SOH            Driver Context       Road Context
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Multimodal Fusion      │
                  │ Battery + Driver + Road│
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Causal Attribution     │
                  │ Engine                 │
                  │                        │
                  │ Battery?              │
                  │ Driver?               │
                  │ Road?                 │
                  └───────────┬────────────┘
                              │
                              ▼
                   Adaptive Protection Logic
                              │
                              ▼
                       Motor / PWM Control
```

The IMU AI contributes:

```text
Vehicle Motion
      ↓
Driving Behavior
      ↓
Context for Cause Attribution
      ↓
Adaptive Protection Decision
```

---

# 35. Example DriveGuard-X Decision Logic

The IMU output will eventually be combined with battery and road information.

Conceptually:

```text
                  Battery Stress Event
                           │
                           ▼
                  Multimodal Analysis
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Battery AI         IMU AI           Road AI
          │                │                │
          ▼                ▼                ▼
       Battery          Driver            Road
       Condition        Context          Context
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                    Cause Attribution
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          Battery        Driver         Road
          Related       Related       Related
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                  Adaptive Protection
```

For example, a future rule-based attribution engine may consider:

```text
High battery stress
+
Aggressive driving context
+
Low road disturbance
        ↓
Likely driver-related event
```

while:

```text
High battery stress
+
Normal driving context
+
High road disturbance
        ↓
Likely road-related event
```

These rules belong to the future **causal attribution layer**, not the IMU classifier itself.

---

# 36. MATLAB/Simulink Integration Plan

The IMU AI model will eventually be integrated into the MATLAB/Simulink environment used for DriveGuard-X system-level simulation.

```text
                    Vehicle Model
                         │
                         ▼
                  Simulated Motion
                         │
                         ▼
                 IMU Measurements
                         │
                         ▼
                  Feature Extraction
                         │
                         ▼
                    IMU AI Model
                         │
                         ▼
                Driving Behavior
                         │
                         ▼
                 Multimodal Fusion
                         │
                         ▼
                 Cause Attribution
                         │
                         ▼
               Adaptive Protection
```

This allows the IMU model to be evaluated as part of the complete DriveGuard-X protection architecture before hardware deployment.

---

# 37. Hardware Deployment Concept

The physical DriveGuard-X prototype is expected to use an ESP32-based architecture.

```text
                 MPU6050
                    │
                    │ I²C
                    ▼
                  ESP32
                    │
                    ▼
             IMU Data Stream
                    │
                    ▼
              4-Sample Window
                    │
                    ▼
             Feature Extraction
                    │
                    ▼
              XGBoost IMU Model
                    │
                    ▼
             Driving Context
```

The model may initially be executed through the software/backend layer rather than directly on the ESP32, depending on final computational and deployment constraints.

---

# 38. Research Findings

### Finding 1: Random splitting can be misleading

An earlier random stratified evaluation produced extremely high performance, including 100% accuracy.

However, because the dataset is generated from overlapping sliding windows, random splitting does not provide a sufficiently realistic estimate of temporal generalization.

Therefore, the final model was evaluated using a leakage-aware temporal split.

### Finding 2: Temporal generalization is considerably harder

The final temporal evaluation produced:

```text
Accuracy = 74.29%
Macro F1 = 74.62%
```

This provides a more conservative and realistic estimate of current model performance.

### Finding 3: Feature reduction improved performance

Reducing the feature set from 60 to 42 features improved accuracy, macro F1, weighted F1, Class 3 recall and Class 3 F1.

### Finding 4: XGBoost provided the strongest baseline

Among the baseline classifiers evaluated using the final temporal split, XGBoost achieved the strongest baseline accuracy and macro F1.

### Finding 5: Class 3 is the primary challenge

The dominant error is:

```text
Class 3 → Class 1
```

This requires further investigation and target-domain validation.

### Finding 6: Temporal distribution shift is significant

Several IMU features changed substantially between earlier and later portions of the dataset.

### Finding 7: The model is a prototype component

The final model is suitable for demonstrating the IMU intelligence component of DriveGuard-X, but it is not a production automotive safety-certified classifier.

---

# 39. Target-Domain Adaptation

The current model is trained using a public dataset.

The next stage is to adapt the model to the actual DriveGuard-X RC vehicle.

```text
Public IMU Dataset
        │
        ▼
Initial IMU Model
        │
        ▼
DriveGuard-X RC Vehicle
        │
        ▼
ESP32 + MPU6050
        │
        ▼
Real Driving Data
        │
        ▼
Data Labelling
        │
        ▼
Target-Domain Dataset
        │
        ▼
Distribution Analysis
        │
        ▼
Model Adaptation / Retraining
        │
        ▼
Validation
        │
        ▼
Real-Time Deployment
```

The public-dataset model therefore represents the **initial IMU model**, not the final hardware-specific model.

---

# 40. Important Limitations

## 40.1 Public Dataset Domain

The final RC vehicle may have different:

- Sensor mounting
- Vehicle dynamics
- Chassis vibration
- Sensor orientation
- Sampling behavior
- Driving patterns

Therefore, public-dataset performance does not guarantee identical performance on the DriveGuard-X RC platform.

## 40.2 Temporal Distribution Shift

The dataset exhibits noticeable temporal changes in several feature distributions.

## 40.3 Class 3 Recall

The final model has relatively low recall for Class 3:

```text
29.85%
```

This is currently the primary classification weakness.

## 40.4 Short Window

The current deployment pipeline uses:

```text
4 IMU samples
```

The suitability of this window size for real-time RC vehicle operation must be experimentally validated.

## 40.5 Sensor Calibration

The final hardware implementation will require calibration of:

- Accelerometer offsets
- Gyroscope offsets
- Sensor orientation
- Sampling rate
- Sensor mounting
- Coordinate-axis conventions

## 40.6 Safety-Critical Usage

The current model is a research prototype.

It should not independently control a real vehicle or safety-critical protection system without extensive validation, fault handling, uncertainty analysis and hardware testing.

---

# 41. What This Model Successfully Demonstrates

The DriveGuard-X IMU AI component demonstrates:

- 6-axis IMU-based driving behavior classification
- Accelerometer and gyroscope feature extraction
- Sliding-window statistical feature engineering
- Leakage-aware temporal evaluation
- Multiple baseline classifier comparison
- XGBoost hyperparameter optimization
- Correlation analysis
- Feature reduction
- Temporal distribution-shift investigation
- Final model selection
- Feature-importance analysis
- Model export and reload verification
- Raw IMU-to-feature pipeline validation
- Preparation for FastAPI integration
- Preparation for MATLAB/Simulink integration
- Preparation for ESP32-based hardware deployment

The primary achievement is the **complete experimental methodology used to develop, evaluate and validate the IMU model for integration into the larger DriveGuard-X system**.

---

# 42. Final Result

```text
Model:
XGBoost Classifier

Input:
6-axis IMU

Window:
4 samples

Original Features:
60

Final Features:
42

Training Samples:
880

Testing Samples:
210

Final Accuracy:
74.29%

Final Macro Precision:
82.18%

Final Macro Recall:
78.74%

Final Macro F1:
74.62%

Final Weighted F1:
72.24%
```

The model has been exported and successfully reloaded, with identical predictions between the original and saved model.

The raw IMU → feature extraction → final 42-feature pipeline has also been numerically validated.

---

# 43. Development Summary

```text
                    BASE RESEARCH PAPER
                            │
                            ▼
                     Public IMU Dataset
                            │
                            ▼
                     Dataset Inspection
                            │
                            ▼
                      Data Quality Check
                            │
                            ▼
                    Raw IMU Verification
                            │
                            ▼
                    Sliding-Window Features
                            │
                            ▼
                       60 Features
                            │
                            ▼
                  Correlation Investigation
                            │
                            ▼
                 Leakage-Aware Temporal Split
                            │
                            ▼
                    Baseline Model Study
                            │
                            ▼
                       XGBoost Selected
                            │
                            ▼
                   Hyperparameter Optimization
                            │
                            ▼
                    Feature Reduction Study
                            │
                     ┌──────┼──────┐
                     ▼      ▼      ▼
                    60     44     42
                     │      │      │
                     └──────┼──────┘
                            ▼
                    42 Features Selected
                            │
                            ▼
                 Temporal Distribution Analysis
                            │
                            ▼
                    Final XGBoost Model
                            │
                            ▼
                    Untouched Test Set
                            │
                            ▼
                     74.29% Accuracy
                            │
                            ▼
                   Model Export & Reload
                            │
                            ▼
                Raw IMU Pipeline Validation
                            │
                            ▼
                    DriveGuard-X IMU AI
```

---

# 44. Future Work

### 1. RC Vehicle Data Collection

Connect the MPU6050 to the ESP32 and collect real driving data from the DriveGuard-X RC vehicle.

### 2. Target-Domain Dataset

Create a labelled dataset from real RC vehicle motion.

### 3. Sensor Calibration

Calibrate:

```text
Accelerometer
Gyroscope
Sensor orientation
Sampling rate
```

### 4. Domain-Shift Analysis

Compare:

```text
Public Dataset
        vs.
RC Vehicle Dataset
```

### 5. Model Adaptation

Retrain or adapt the model using target-domain RC data.

### 6. Real-Time Inference

Implement:

```text
MPU6050
   ↓
ESP32
   ↓
4-Sample Window
   ↓
42 Features
   ↓
IMU Model
   ↓
Driving Context
```

### 7. FastAPI Integration

Expose the model through the DriveGuard-X backend.

### 8. MATLAB/Simulink Integration

Use the IMU model within the system-level simulation environment.

### 9. Multimodal Fusion

Combine:

```text
Battery AI
+
IMU AI
+
Road AI
```

### 10. Causal Attribution

Estimate whether abnormal battery-stress events are primarily:

```text
Battery-related
Driver-related
Road-related
Mixed
```

### 11. Adaptive Protection

Use the attribution result to determine the appropriate protection response.

---

# 45. Complete DriveGuard-X Vision

The IMU AI is ultimately one part of the larger DriveGuard-X architecture.

```text
                         ┌───────────────────────┐
                         │      DRIVEGUARD-X     │
                         │ Adaptive EV Protection│
                         └───────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
       ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
       │   Battery   │        │     IMU     │        │    Road     │
       │   Sensors   │        │   Sensors   │        │   Sensors   │
       └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
              │                      │                      │
              ▼                      ▼                      ▼
       ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
       │  Battery AI │        │   Driving   │        │   Road AI   │
       │    SOH      │        │     AI      │        │ Road Event  │
       └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Multimodal Fusion     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Causal Attribution    │
                         │ Engine                │
                         │                       │
                         │ Battery?              │
                         │ Driver?               │
                         │ Road?                 │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Adaptive Protection   │
                         │ Logic                 │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Motor / PWM Control   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ DriveGuard-X Hardware │
                         │ Prototype             │
                         └───────────────────────┘
```

---

# 46. Conclusion

The DriveGuard-X IMU AI module began with a published machine-learning methodology for identifying driving behavior from inertial measurements and progressively developed into a leakage-aware, deployment-oriented classification pipeline.

The model uses 6-axis IMU measurements from accelerometer and gyroscope channels and converts short 4-sample windows into statistical features.

An initial 60-feature representation was investigated and subsequently reduced to 42 features. Multiple machine-learning models were compared, with XGBoost providing the strongest baseline performance.

A temporal, leakage-aware evaluation strategy was adopted to avoid overly optimistic results caused by overlapping sliding windows.

The final 42-feature XGBoost model achieved:

```text
Accuracy        = 74.29%
Macro Precision = 82.18%
Macro Recall    = 78.74%
Macro F1        = 74.62%
Weighted F1     = 72.24%
```

on an untouched temporal test set containing 210 samples.

The exported model was successfully reloaded with identical predictions, and the raw IMU-to-42-feature reconstruction pipeline was validated to numerical precision.

The current model is suitable as a **research-grade IMU AI component for the DriveGuard-X prototype**.

Its main remaining challenge is target-domain adaptation to the actual RC vehicle, particularly the separation of Class 3 from Class 1.

The next stage is to collect real ESP32 + MPU6050 data, evaluate domain shift, adapt the model and integrate the resulting driving-context information with the Battery AI and Road AI modules.

The ultimate objective is to use these multimodal outputs for **cause-aware adaptive battery protection**, rather than relying solely on fixed battery thresholds.
