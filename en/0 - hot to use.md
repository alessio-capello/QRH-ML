# 0. HOW TO USE THIS QRH

**Quick navigation**
* [1. Decision matrix](#1-decision-matrix)
* [2.1 Baseline ML & Linear Methods](2.1%20-%20baseline_lineari.md)
* [2.2 Ensemble Methods](2.2%20-%20ensemble.md)
* [2.3 Unsupervised & Representation](2.3%20-%20unsupervised.md)
* [2.4 Deep Learning](2.4%20-%20deep%20learning.md)
* [2.5 Time Series](2.5%20-%20time%20series.md)
* [2.6 Computer Vision](2.6%20-%20computer%20vision.md)
* [2.7 NLP](2.7%20-%20NLP.md)
* [2.8 Reinforcement Learning](2.8%20-%20reinforcement%20learning.md)
* [3. Troubleshooting](3%20-%20troubleshooting.md)

**GRAMMAR AND TAXONOMY**
This handbook is designed for rapid triage. The header of each model card contains Operational Tags enclosed in square brackets for immediate profiling:
* **Paradigm:** `[SUPERVISED]`, `[UNSUPERVISED]`, `[SELF-SUPERVISED]`, `[RL]`.
* **Data Domain:** `[TABULAR]`, `[VISION]`, `[TEXT]`, `[TIME SERIES]`, `[SPATIAL]`.
* **Task:** `[CLASSIFICATION]`, `[REGRESSION]`, `[CLUSTERING]`, `[GENERATION]`, `[ANOMALY DETECTION]`.
* **Warning / Status:** `[HIGH INTERPRETABILITY]`, `[SOTA]` (State Of The Art), `[BLACK-BOX]`, `[MEMORY INTENSIVE]`.

**ML PARADIGMS AND TASK TYPES**
* **Supervised learning:** Learns from labeled examples, where the target outcome is known; commonly used for regression and classification.
* **Unsupervised learning:** Finds structure in unlabeled data, such as clusters, latent representations, or anomalies.
* **Self-supervised learning:** Creates training signals from the data itself, for example by masking part of an input and predicting it.
* **Semi-supervised learning:** Combines a small labeled dataset with a larger unlabeled dataset to improve learning.
* **Reinforcement learning (RL):** An agent learns by taking actions in an environment and optimizing cumulative rewards.
* **Regression:** Predicts a continuous numerical value, such as a price, temperature, or demand.
* **Classification:** Predicts a discrete class or probability, such as spam/not spam or a disease category.

**READING STANDARD: GO / NO-GO**
* **GO:** Optimal conditions in which the model shines or represents the de facto industrial baseline. If your problem matches the GO, stop and implement it.
* **NO-GO:** Conditions in which using the model will cause engineering bottlenecks, mathematical errors (e.g. OOM), or catastrophic performance. Ignoring a NO-GO means failing deployment.

**ESSENTIAL ACRONYM GLOSSARY**
* **OOM:** [Out Of Memory](#essential-acronym-glossary) (RAM or GPU VRAM exhaustion during training).
* **BCE / CCE:** [Binary / Categorical Cross-Entropy](#essential-acronym-glossary) (Loss functions for classification).
* **MSE / MAE:** [Mean Squared / Absolute Error](#essential-acronym-glossary) (Loss functions for regression).
* **SOTA:** [State Of The Art](#essential-acronym-glossary) (The model currently performing best for a given task).
* **VRAM:** [Video RAM](#essential-acronym-glossary) (GPU memory, the primary physical limit in Deep Learning).
* **KL:** [Kullback-Leibler Divergence](#essential-acronym-glossary) (Measure of difference between probability distributions).
* **CPU / GPU:** [Central / Graphics Processing Unit](#essential-acronym-glossary) (CPU = general-purpose compute; GPU = parallel compute for ML workloads).
* **ML / AI / DL:** [Machine Learning / Artificial Intelligence / Deep Learning](#essential-acronym-glossary) (core ML terminology used throughout the handbook).
* **NLP / RL:** [Natural Language Processing / Reinforcement Learning](#essential-acronym-glossary) (the main task families referenced in the model cards).
* **PCA / KNN / SVM / MLP:** [Principal Component Analysis / K-Nearest Neighbors / Support Vector Machines / Multi-Layer Perceptron](#essential-acronym-glossary) (common baseline and classical ML methods).
* **CNN / RNN / LSTM / GRU / ViT / LLM:** [Convolutional Neural Network / Recurrent Neural Network / Long Short-Term Memory / Gated Recurrent Unit / Vision Transformer / Large Language Model](#essential-acronym-glossary) (deep learning architectures used in the handbook).
* **VAE / GAN:** [Variational Autoencoder / Generative Adversarial Network](#essential-acronym-glossary) (representative generative modeling families).
* **TF-IDF / BERT / T5:** [Term Frequency–Inverse Document Frequency / Bidirectional Encoder Representations from Transformers / Text-To-Text Transfer Transformer](#essential-acronym-glossary) (core NLP representations and model families).

> Most acronyms in this handbook are linked to this glossary for quick lookup.

---

# 1. DECISION MATRIX

This section acts as a dispatcher. Identify the nature of your input and your operational constraints to be redirected to the correct model card.

## 1.1 DOMAIN / TASK MATRIX (PRIMARY TRIAGE)
Cross the structure of the input tensor (Rows) with the macro algorithmic objective (Columns).

| DATA DOMAIN | PREDICTIVE (Class. / Regr.) | DISCOVERY (Clustering / Dim. Red.) | ANOMALY DETECTION & REPRESENTATION | COMPLEX / GENERATIVE TASKS |
| :--- | :--- | :--- | :--- | :--- |
| **Tabular** | XGBoost / LightGBM / RF | K-Means / PCA / UMAP | DBSCAN / Dense Autoencoders | VAE (Synthetic Data Generation) |
| **Images (2D/3D)** | ResNet / EfficientNet / ViT | UMAP (on extracted features) | Conv-Autoencoders (Denoising/Anomalies) | YOLO / U-Net / Mask R-CNN |
| **Time Series / Signals**| ARIMA / Prophet / 1D-CNN | Dynamic Time Warping + K-Means | 1D-CNN AE / LSTM-Autoencoders | LSTM / GRU (Seq-to-Seq) |
| **Text (NLP)** | TF-IDF / BERT / RoBERTa | LDA / UMAP (on TF-IDF) | Masked Language Modeling | LLMs (Llama, Qwen) / T5 |
| **Interactive Environment**| DQN (Discrete Actions) | N/A | N/A | PPO / SAC (Continuous Actions) |

## 1.2 DECISION TREE: TABULAR DATA
Around 70% of business tasks fall here. Follow the boolean logic to filter the models.

* **[IF]** Full mathematical interpretability and/or legal explainability of the decision is required:
  * **[AND IF]** Features are continuous and independent — [**Linear / Logistic Regression**](2.1%20-%20baseline_lineari.md#211-linear--logistic-regression)
  * **[AND IF]** The logic is based on rigid business rules (If-Then) — [**Decision Trees**](2.1%20-%20baseline_lineari.md#213-decision-trees-cart)
* **[IF]** Full interpretability is not required (Black-Box allowed):
  * **[AND IF]** There is almost no time for hyper-tuning and a robust out-of-the-box baseline is needed — [**Random Forest**](2.2%20-%20ensemble.md#221-random-forest)
  * **[AND IF]** You are looking for the maximum absolute predictive performance (SOTA) — [**XGBoost / LightGBM**](2.2%20-%20ensemble.md#222-gradient-boosting-machines-xgboost-lightgbm-catboost)
* **[IF]** Pure anomaly detection on unlabeled tabular data:
  * **[AND IF]** Features are linear or density-based — [**DBSCAN**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan)
  * **[AND IF]** The dataset has highly complex, non-linear patterns — [**Dense Autoencoders**](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae)
* **[IF]** The dataset is in the Terabyte range (well beyond system RAM):
  * — [**Multi-Layer Perceptron (MLP)**](2.4%20-%20deep%20learning.md#241-multi-layer-perceptron-mlp--ffn) processed in mini-batches on GPU.

## 1.3 DECISION TREE: VISION AND TIME SERIES (SPACE & TIME)

**COMPUTER VISION (3D/4D Tensors)**
* **[IF]** The task is to find unlabeled morphological anomalies (e.g. industrial defectoscopy on perfect parts) — [**Conv-Autoencoders**](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae) (trained only on healthy samples, reconstruct defects poorly).
* **[IF]** The task is to identify what is in the image (Classification) — [**ResNet / EfficientNet**](2.6%20-%20computer%20vision.md#261-2d-cnn-backbones-resnet-efficientnet) (or [**ViT**](2.6%20-%20computer%20vision.md#265-vision-transformers-vit) if the dataset is massive).
* **[IF]** The task is to find the exact position of the objects:
  * **[AND IF]** Real-time localization with rectangular bounding boxes is required — [**YOLO**](2.6%20-%20computer%20vision.md#262-yolo-family-you-only-look-once)
  * **[AND IF]** Millimetric classification of each individual pixel is required (no overlap) — [**U-Net**](2.6%20-%20computer%20vision.md#263-u-net)
  * **[AND IF]** Individual objects of the same class overlapping each other must be separated — [**Mask R-CNN**](2.6%20-%20computer%20vision.md#264-mask-r-cnn)

**TIME SERIES & SIGNALS (Temporal Tensors)**
* **[IF]** The task is to detect anomalous behaviors in the signal over time (vibrations, EEG, network logs) — [**1D-CNN Autoencoder**](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae) (fast) or [**LSTM Autoencoder**](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae) (long memory).
* **[IF]** Univariate Forecasting (Predict the future of a single historical variable):
  * **[AND IF]** Strong short-term autoregressive logic — [**ARIMA / SARIMAX**](2.5%20-%20time%20series.md#251-arima--sarimax)
  * **[AND IF]** Strong combined seasonality (e.g. retail sales) — [**Prophet**](2.5%20-%20time%20series.md#252-prophet-meta)
* **[IF]** Multivariate Forecasting or high-frequency Pattern Recognition:
  * **[AND IF]** Need for GPU parallelization and extreme speed — [**1D-CNN / TCN**](2.5%20-%20time%20series.md#253-1d-cnn--tcn-temporal-convolutional-networks)
  * **[AND IF]** Complex variable-length temporal relations — [**LSTM / GRU**](2.5%20-%20time%20series.md#254-rnn-lstm--gru-recurrent-neural-networks)

## 1.4 DECISION TREE: NLP & TEXT GENERATION
* **[IF]** The task is to generate long, dynamic text output, or act as an assistant/agent — [**LLM Decoder-only (Llama, Qwen)**](2.7%20-%20NLP.md#273-transformer-decoders-llms-llama-qwen-etc)
* **[IF]** The task is the direct structured transformation of text (Translation, Summarization) — [**Sequence-to-Sequence (T5, BART)**](2.7%20-%20NLP.md#274-sequence-to-sequence-t5-bart)
* **[IF]** The task is to label text (Sentiment, NER) or extract dense vector embeddings:
  * **[AND IF]** Deep semantic understanding of the language is required — [**Transformer Encoders (BERT, RoBERTa)**](2.7%20-%20NLP.md#272-transformer-encoders-bert-roberta)
  * **[AND IF]** An ultra-light baseline computable on CPU in a few seconds is required — [**TF-IDF + Naive Bayes**](2.7%20-%20NLP.md#271-tf-idf--naive-bayes)

## 1.5 DECISION TREE: UNSUPERVISED, CLUSTERING & REPRESENTATION
This section maps models when labels are absent (unlabeled data) and the goal is pattern discovery or data compression (Feature Extraction).

* **[IF]** The task is pure clustering on native tabular data:
  * **[AND IF]** Spherical clusters are assumed and the number of groups is known a priori — [**K-Means**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan)
  * **[AND IF]** Arbitrary shapes are sought, noise should be isolated, and the number of groups is unknown — [**DBSCAN**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan)
* **[IF]** The dataset is complex (Images, high-frequency IoT signals, Text) and direct clustering would fail due to the Curse of Dimensionality (Deep Clustering Pipeline):
  * **[AND IF]** The goal is to visually group data in 2D/3D — Feature extraction + [**UMAP / t-SNE**](2.3%20-%20unsupervised.md#232-dimensionality-reduction-pca-t-sne-umap) + [**DBSCAN**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan)
  * **[AND IF]** The goal is logical high-dimensional grouping (e.g. 64-128D) — [**Autoencoder (Conv/1D/Dense)**](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae) to extract the Latent Space — [**K-Means / DBSCAN**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan) on the Latent Space.
* **[IF]** The task is Dimensionality Reduction to speed up downstream Machine Learning models:
  * **[AND IF]** You want to preserve global linear distances rigorously — [**PCA**](2.3%20-%20unsupervised.md#232-dimensionality-reduction-pca-t-sne-umap)
  * **[AND IF]** You want to preserve local topological relations (non-linear manifolds) — [**UMAP**](2.3%20-%20unsupervised.md#232-dimensionality-reduction-pca-t-sne-umap)

## 1.6 HARDWARE & LATENCY CONSTRAINTS MATRIX
The final check before starting training.

| PROJECT CONSTRAINTS | ACTION / RECOMMENDED MODEL | MODELS TO AVOID (NO-GO) |
| :--- | :--- | :--- |
| **Ultra-Low Inference Latency (Milliseconds)** | [Linear/Logistic](2.1%20-%20baseline_lineari.md#211-linear--logistic-regression), [Decision Trees](2.1%20-%20baseline_lineari.md#213-decision-trees-cart), [YOLO](2.6%20-%20computer%20vision.md#262-yolo-family-you-only-look-once), [TF-IDF](2.7%20-%20NLP.md#271-tf-idf--naive-bayes), [1D-CNN](2.5%20-%20time%20series.md#253-1d-cnn--tcn-temporal-convolutional-networks). | [KNN](2.1%20-%20baseline_lineari.md#212-k-nearest-neighbors-knn) (O(N) in prediction), [RF](2.2%20-%20ensemble.md#221-random-forest) (>500 trees), [Mask R-CNN](2.6%20-%20computer%20vision.md#264-mask-r-cnn), [LLMs](2.7%20-%20NLP.md#273-transformer-decoders-llms-llama-qwen-etc). |
| **Edge Serverless Deployment (No GPU available)** | [Random Forest](2.2%20-%20ensemble.md#221-random-forest), [XGBoost](2.2%20-%20ensemble.md#222-gradient-boosting-machines-xgboost-lightgbm-catboost) (CPU inference), [SVM](2.1%20-%20baseline_lineari.md#214-support-vector-machines-svm), [Prophet](2.5%20-%20time%20series.md#252-prophet-meta). | [Transformers](2.7%20-%20NLP.md#272-transformer-encoders-bert-roberta), [ViT](2.6%20-%20computer%20vision.md#265-vision-transformers-vit), [Deep CNNs](2.6%20-%20computer%20vision.md#261-2d-cnn-backbones-resnet-efficientnet), [LSTM](2.5%20-%20time%20series.md#254-rnn-lstm--gru-recurrent-neural-networks), [Generative Autoencoders](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae). |
| **Very Limited GPU VRAM (< 8 GB)** | [1D-CNN](2.5%20-%20time%20series.md#253-1d-cnn--tcn-temporal-convolutional-networks), [GRU](2.5%20-%20time%20series.md#254-rnn-lstm--gru-recurrent-neural-networks), [ResNet-50](2.6%20-%20computer%20vision.md#261-2d-cnn-backbones-resnet-efficientnet) (batch < 16), [YOLO Nano](2.6%20-%20computer%20vision.md#262-yolo-family-you-only-look-once), [LoRA on LLM](2.7%20-%20NLP.md#273-transformer-decoders-llms-llama-qwen-etc). | [Mask R-CNN](2.6%20-%20computer%20vision.md#264-mask-r-cnn), [ViT](2.6%20-%20computer%20vision.md#265-vision-transformers-vit), [LLM Full-Parameter](2.7%20-%20NLP.md#273-transformer-decoders-llms-llama-qwen-etc), [High-Resolution 3D U-Net](2.6%20-%20computer%20vision.md#263-u-net). |
| **Cold Start / Small Data (Very Little Data)** | [SVM](2.1%20-%20baseline_lineari.md#214-support-vector-machines-svm), [Decision Trees](2.1%20-%20baseline_lineari.md#213-decision-trees-cart), [Naive Bayes](2.7%20-%20NLP.md#271-tf-idf--naive-bayes). *(For Vision/Text: Transfer Learning is mandatory).* | [Tabular MLPs](2.4%20-%20deep%20learning.md#241-multi-layer-perceptron-mlp--ffn), [ViT trained from scratch](2.6%20-%20computer%20vision.md#265-vision-transformers-vit), [LSTM](2.5%20-%20time%20series.md#254-rnn-lstm--gru-recurrent-neural-networks), [Generative GANs/VAE](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae). |
