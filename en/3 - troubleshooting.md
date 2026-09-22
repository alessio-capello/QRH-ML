# 3. TROUBLESHOOTING CHECKLISTS

**Quick links**
* [0. How to use this QRH](0%20-%20hot%20to%20use.md#1-decision-matrix)
* [2.1 Baseline ML & Linear Methods](2.1%20-%20baseline_lineari.md)
* [2.3 Unsupervised & Representation](2.3%20-%20unsupervised.md)
* [2.7 NLP](2.7%20-%20NLP.md)
* [2.8 Reinforcement Learning](2.8%20-%20reinforcement%20learning.md)
* Sections: [Loss = NaN](#-symptom-loss--nan-not-a-number), [Flat Loss](#-symptom-flat-loss-from-the-first-epoch-the-model-does-not-learn), [Underfitting](#-symptom-high-loss-in-training-high-loss-in-validation-severe-underfitting), [Overfitting](#-symptom-low-loss-in-training-validation-loss-diverges-or-rises-chronic-overfitting), [OOM](#-symptom-oom-out-of-memory-error-on-gpu)

This section contains emergency procedures for diagnosing and resolving the most common bottlenecks during a model’s lifecycle (Training, Data Pipeline, Production).

## 3.1 TRAINING FAILURES (LOSS & GRADIENTS)
Procedures for when training fails mathematically or stalls.

### [SYMPTOM] Loss = NaN (Not a Number)
* **Diagnosis 1: Gradient Explosion.** Gradients have grown exponentially, surpassing the float32 limit. Typical in RNNs, LSTMs, and Transformers without warm-up.
  * *Quick Fix:* Add Gradient Clipping (`clipnorm=1.0` or `clipvalue=0.5`). Reduce the `learning_rate` drastically.
* **Diagnosis 2: Corrupted Input (Division by Zero / Log of Zero).**
  * *Quick Fix:* Check the dataset for unseen `NaN` or `Inf` values. Add an epsilon (`+ 1e-8`) before logarithmic operations or divisions in loss calculations or custom functions.
* **Diagnosis 3: Excessive Learning Rate.**
  * *Quick Fix:* Reduce the LR by one order of magnitude (e.g. from 1e-3 to 1e-4).

### [SYMPTOM] Flat Loss from the First Epoch (The model does not learn)
* **Diagnosis 1: Vanishing Gradients.** The error signal fades out as it propagates backward (typical in deep networks with Sigmoid/Tanh activations).
  * *Quick Fix:* Replace activations with `ReLU`, `LeakyReLU`, or `Mish`. Implement normalization (`Batch Normalization` or `Layer Normalization`). Check weight initialization (use `He / Glorot`).
* **Diagnosis 2: Input Scale Discrepancy.** Input data have completely different magnitudes (e.g. one feature varies from 0 to 1, another from 1,000 to 10,000).
  * *Quick Fix:* Stop training. Apply `StandardScaler` or `MinMaxScaler` to all numerical features.

---

## 3.2 PERFORMANCE ISSUES (UNDERFITTING & OVERFITTING)
Procedures for when the model completes training but the metrics are wrong.

### [SYMPTOM] High Loss in Training, High Loss in Validation (Severe Underfitting)
* **Diagnosis:** The model lacks the capacity (parameters) to map the complexity of the problem, or is overly regularized.
  * *Quick Fix 1 (Capacity):* Increase complexity (more trees in RF, more layers/neurons in MLP, switch from ResNet-18 to ResNet-50).
  * *Quick Fix 2 (Regularization):* Reduce Dropout, lower L1/L2 penalties (Weight Decay), decrease `min_samples_leaf` in trees.
  * *Quick Fix 3 (Features):* Do Feature Engineering (create polynomial interactions).

### [SYMPTOM] Low Loss in Training, Validation Loss Diverges or Rises (Chronic Overfitting)
* **Diagnosis:** The model is memorizing the training data (noise included) and failing to generalize to new data.
  * *Quick Fix 1 (Early Stopping):* Stop training at the elbow point before validation loss starts rising.
  * *Quick Fix 2 (Data):* Increase the dataset or use heavy Data Augmentation (images/audio).
  * *Quick Fix 3 (Regularization):* Increase Dropout (e.g. 0.5), add Weight Decay (L2), reduce model depth (`max_depth` for trees, reduce layers for neural networks).

---

## 3.3 DATA PIPELINE & HARDWARE FAILURES
Procedures for infrastructure issues or logical contamination of data.

### [SYMPTOM] OOM (Out Of Memory) Error on GPU
* **Diagnosis:** The gradient tensor and feature maps exceed physical VRAM (typical in Transformers or high-resolution Computer Vision).
  * *Quick Fix 1 (Immediate):* Iteratively halve the `batch_size` (e.g. 64 -> 32 -> 16).
  * *Quick Fix 2 (Gradient Accumulation):* If a low `batch_size` makes gradients unstable, simulate larger batches by calculating gradients on small batches and updating weights only every N steps.
  * *Quick Fix 3 (Precision):* Enable Mixed Precision Training (FP16 / BF16). It halves VRAM usage with almost no loss in accuracy.

### [SYMPTOM] Model is “Perfect” in Test (e.g. Accuracy 99%) but a disaster in the real world
* **Diagnosis: Data Leakage.** Target information is accidentally included in the features, or the Validation Set is contaminated by the Training Set.
  * *Quick Fix 1 (Time Series):* Check whether a strict temporal split was used. Never use purely random splits (`train_test_split`) on time series.
  * *Quick Fix 2 (Scaling):* Check that `StandardScaler.fit()` was applied ONLY to the Training set, not to the entire primary dataset.
  * *Quick Fix 3 (Implicit Features):* Ensure no feature is a direct consequence of the target variable (e.g. using “days of hospitalization” to predict whether the patient will be admitted).

### [SYMPTOM] Extremely high accuracy (99%), but Precision/Recall for one class are zero
* **Diagnosis: Severe Class Imbalance.** (e.g. 99% of transactions are legitimate, 1% are fraud. Predicting “Legitimate” always gives 99% accuracy at zero cost).
  * *Quick Fix 1 (Metrics):* Stop looking at Accuracy. Use F1-Score, Precision, Recall, and AUC-PR.
  * *Quick Fix 2 (Algorithmic):* Use weighted loss functions (Focal Loss, Class Weights in the algorithm).
  * *Quick Fix 3 (Data):* Oversample the minority class (SMOTE) or undersample the majority class (only in the training set).

---

## 3.4 INFERENCE & PRODUCTION
Procedures for monitoring and performance in deployment.

### [SYMPTOM] Inference latency is too high (API timeouts or delays)
* **Diagnosis:** The saved model (in raw format, e.g. PyTorch `.pt` or `.pkl`) performs unnecessary calculations (e.g. gradient graphs) or is not optimized for the target hardware.
  * *Quick Fix 1:* Export the model in compiled ONNX or TensorRT formats for hardware acceleration.
  * *Quick Fix 2:* Post-training quantization. Convert model weights from Float32 to INT8 (speed up to 4x, minimal precision loss).
  * *Quick Fix 3:* Batch concurrent API requests in micro-batches (Dynamic Batching) instead of processing them one at a time.

### [SYMPTOM] Performance gradually degrades after N months in production
* **Diagnosis: Concept / Data Drift.** User habits (Concept) or the physical format of incoming sensors (Data) have changed relative to the historical training data.
  * *Quick Fix (Monitoring):* Implement statistical tests (e.g. Kolmogorov-Smirnov Test, PSI - Population Stability Index) on the input distribution in production.
  * *Quick Fix (Resolution):* Schedule a retraining cycle or switch to a Continual Learning paradigm including fresh data samples from the last 30 days (giving them more algorithmic weight).
