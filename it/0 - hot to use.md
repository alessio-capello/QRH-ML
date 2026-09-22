# 0. HOW TO USE THIS QRH

**Quick navigation**
* [1. Decision matrix](#1-decision-matrix)
* [2.1 Baseline ML & Metodi Lineari](2.1%20-%20baseline_lineari.md)
* [2.2 Ensemble Methods](2.2%20-%20ensemble.md)
* [2.3 Unsupervised & Rappresentazione](2.3%20-%20unsupervised.md)
* [2.4 Deep Learning](2.4%20-%20deep%20learning.md)
* [2.5 Time Series](2.5%20-%20time%20series.md)
* [2.6 Computer Vision](2.6%20-%20computer%20vision.md)
* [2.7 NLP](2.7%20-%20NLP.md)
* [2.8 Reinforcement Learning](2.8%20-%20reinforcement%20learning.md)
* [3. Troubleshooting](3%20-%20troubleshooting.md)

**GRAMMATICA E TASSONOMIA**
Questo manuale è progettato per il triage rapido. L'intestazione di ogni scheda modello contiene Tag Operativi racchiusi tra parentesi quadre per un'immediata profilazione:
* **Paradigma:** `[SUPERVISIONATO]`, `[NON SUPERVISIONATO]`, `[SELF-SUPERVISIONATO]`, `[RL]`.
* **Dominio Dato:** `[TABULARE]`, `[VISIONE]`, `[TESTO]`, `[TIME SERIES]`, `[SPAZIALE]`.
* **Task:** `[CLASSIFICAZIONE]`, `[REGRESSIONE]`, `[CLUSTERING]`, `[GENERAZIONE]`, `[ANOMALY DETECTION]`.
* **Warning / Status:** `[ALTA INTERPRETABILITA']`, `[SOTA]` (State Of The Art), `[BLACK-BOX]`, `[MEMORY INTENSIVE]`.

**STANDARD DI LETTURA: GO / NO-GO**
* **GO:** Condizioni ottimali in cui il modello brilla o rappresenta la baseline industriale de facto. Se il tuo problema matcha il GO, fermati e implementa.
* **NO-GO:** Condizioni in cui l'uso del modello causerà colli di bottiglia ingegneristici, errori matematici (es. OOM) o prestazioni catastrofiche. Ignorare un NO-GO significa fallire il deployment.

**GLOSSARIO ACRONIMI ESSENZIALI**
* **OOM:** [Out Of Memory](#glossario-acronimi-essenziali) (Esaurimento della RAM o VRAM della GPU durante l'addestramento).
* **BCE / CCE:** [Binary / Categorical Cross-Entropy](#glossario-acronimi-essenziali) (Funzioni di loss per classificazione).
* **MSE / MAE:** [Mean Squared / Absolute Error](#glossario-acronimi-essenziali) (Funzioni di loss per regressione).
* **SOTA:** [State Of The Art](#glossario-acronimi-essenziali) (Il modello attualmente più performante per un dato task).
* **VRAM:** [Video RAM](#glossario-acronimi-essenziali) (Memoria della GPU, limite fisico primario nel Deep Learning).
* **KL:** [Kullback-Leibler Divergence](#glossario-acronimi-essenziali) (Misura di differenza tra distribuzioni di probabilità).

---

# 1. DECISION MATRIX

Questa sezione funge da dispatcher. Identifica la natura del tuo input e le tue limitazioni operative per essere re-indirizzato alla scheda modello corretta.

## 1.1 MATRICE DOMINIO / TASK (TRIAGE PRIMARIO)
Incrocia la struttura del tensore di input (Righe) con il macro-obiettivo algoritmico (Colonne).

| DOMINIO DATO | PREDITTIVO (Class. / Regr.) | DISCOVERY (Clustering / Dim. Red.) | ANOMALY DETECTION & RAPPRESENTAZIONE | TASK COMPLESSI / GENERATIVI |
| :--- | :--- | :--- | :--- | :--- |
| **Tabulare** | XGBoost / LightGBM / RF | K-Means / PCA / UMAP | DBSCAN / Dense Autoencoders | VAE (Generazione Dati Sintetici) |
| **Immagini (2D/3D)** | ResNet / EfficientNet / ViT | UMAP (su feature estratte) | Conv-Autoencoders (Denoising/Anomalie) | YOLO / U-Net / Mask R-CNN |
| **Time Series / Segnali**| ARIMA / Prophet / 1D-CNN | Dynamic Time Warping + K-Means | 1D-CNN AE / LSTM-Autoencoders | LSTM / GRU (Seq-to-Seq) |
| **Testo (NLP)** | TF-IDF / BERT / RoBERTa | LDA / UMAP (su TF-IDF) | Masked Language Modeling | LLMs (Llama, Qwen) / T5 |
| **Ambiente Interattivo**| DQN (Azioni Discrete) | N/A | N/A | PPO / SAC (Azioni Continue) |

## 1.2 ALBERO DECISIONALE: DATI TABULARI
Il 70% dei task aziendali ricade qui. Segui la logica booleana per filtrare i modelli.

* **[IF]** È richiesta interpretabilità matematica totale e/o spiegabilità legale della decisione:
  * **[AND IF]** Le feature sono continue e indipendenti ➡️ **Linear / Logistic Regression**
  * **[AND IF]** La logica è basata su regole di business rigide (If-Then) ➡️ **Decision Trees**
* **[IF]** Non è richiesta interpretabilità totale (Black-Box ammessa):
  * **[AND IF]** Il tempo per l'iper-tuning è quasi nullo, serve una baseline robusta out-of-the-box ➡️ **Random Forest**
  * **[AND IF]** Si cerca la massima performance predittiva assoluta (SOTA) ➡️ **XGBoost / LightGBM**
* **[IF]** Anomaly Detection puro su dati tabulari non etichettati:
  * **[AND IF]** Le feature sono lineari o basate su densità ➡️ **DBSCAN**
  * **[AND IF]** Il dataset ha pattern altamente complessi e non lineari ➡️ **Dense Autoencoders**
* **[IF]** Il dataset è nell'ordine dei Terabyte (esula dalla RAM di sistema):
  * ➡️ **Multi-Layer Perceptron (MLP)** processato a mini-batch su GPU.

## 1.3 ALBERO DECISIONALE: VISIONE E TIME SERIES (SPAZIO & TEMPO)

**COMPUTER VISION (Tensori 3D/4D)**
* **[IF]** Il task è trovare anomalie morfologiche non etichettate (es. difettoscopia industriale su pezzi perfetti) ➡️ **Conv-Autoencoders** (addestrati solo su campioni sani, ricostruiscono male i difetti).
* **[IF]** Il task è identificare cosa c'è nell'immagine (Classificazione) ➡️ **ResNet / EfficientNet** (o **ViT** se dataset massivo).
* **[IF]** Il task è trovare la posizione esatta degli oggetti:
  * **[AND IF]** Serve localizzazione real-time con bounding box rettangolari ➡️ **YOLO**
  * **[AND IF]** Serve classificazione millimetrica di ogni singolo pixel (no overlap) ➡️ **U-Net**
  * **[AND IF]** Bisogna separare singoli oggetti della stessa classe sovrapposti ➡️ **Mask R-CNN**

**TIME SERIES & SEGNALI (Tensori Temporali)**
* **[IF]** Il task è individuare comportamenti anomali nel segnale nel corso del tempo (vibrazioni, EEG, log di rete) ➡️ **1D-CNN Autoencoder** (veloce) o **LSTM Autoencoder** (memoria lunga).
* **[IF]** Univariate Forecasting (Predire il futuro di una singola variabile storicizzata):
  * **[AND IF]** Forte logica autoregressiva a breve termine ➡️ **ARIMA / SARIMAX**
  * **[AND IF]** Forti stagionalità combinate (es. vendite retail) ➡️ **Prophet**
* **[IF]** Multivariate Forecasting o Pattern Recognition ad alta frequenza:
  * **[AND IF]** Necessità di parallelizzazione su GPU e velocità estrema ➡️ **1D-CNN / TCN**
  * **[AND IF]** Relazioni temporali complesse a lunghezza variabile ➡️ **LSTM / GRU**

## 1.4 ALBERO DECISIONALE: NLP & GENERAZIONE TESTO
* **[IF]** Il task è generare un output testuale lungo, dinamico, o agire come assistente/agente ➡️ **LLM Decoder-only (Llama, Qwen)**
* **[IF]** Il task è la trasformazione strutturata e diretta del testo (Traduzione, Summarization) ➡️ **Sequence-to-Sequence (T5, BART)**
* **[IF]** Il task è etichettare il testo (Sentiment, NER) o estrarre embeddings vettoriali densi:
  * **[AND IF]** Serve comprensione profonda della semantica del linguaggio ➡️ **Transformer Encoders (BERT, RoBERTa)**
  * **[AND IF]** Serve una baseline ultra-leggera calcolabile su CPU in pochi secondi ➡️ **TF-IDF + Naive Bayes**

## 1.5 ALBERO DECISIONALE: UNSUPERVISED, CLUSTERING & RAPPRESENTAZIONE
Questa sezione mappa i modelli quando NON si hanno label (dati non etichettati) e l'obiettivo è la scoperta di pattern o la compressione del dato (Feature Extraction).

* **[IF]** Il task è il Clustering puro su dati tabulari nativi:
  * **[AND IF]** Si ipotizzano cluster sferici e si conosce a priori il numero di gruppi ➡️ [**K-Means**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan)
  * **[AND IF]** Si cercano forme arbitrarie, si vuole isolare il rumore e non si conosce il numero di gruppi ➡️ [**DBSCAN**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan)
* **[IF]** Il dataset è complesso (Immagini, Segnali IoT ad alta frequenza, Testo) e il Clustering diretto fallirebbe per la Maledizione della Dimensionalità (Deep Clustering Pipeline):
  * **[AND IF]** L'obiettivo è raggruppare visivamente i dati in 2D/3D ➡️ Estrazione feature + [**UMAP / t-SNE**](2.3%20-%20unsupervised.md#232-dimensionality-reduction-pca-t-sne-umap) + [**DBSCAN**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan)
  * **[AND IF]** L'obiettivo è un raggruppamento logico ad alta dimensione (es. 64-128D) ➡️ [**Autoencoder (Conv/1D/Dense)**](2.3%20-%20unsupervised.md#233-autoencoders-standard--vae) per estrarre lo Spazio Latente ➡️ [**K-Means / DBSCAN**](2.3%20-%20unsupervised.md#231-clustering-k-means--dbscan) sullo Spazio Latente.
* **[IF]** Il task è la Riduzione della Dimensionalità per accelerare modelli di Machine Learning a valle:
  * **[AND IF]** Si vogliono preservare rigorosamente le distanze globali e lineari ➡️ [**PCA**](2.3%20-%20unsupervised.md#232-dimensionality-reduction-pca-t-sne-umap)
  * **[AND IF]** Si vogliono preservare le relazioni topologiche locali (manifold non lineari) ➡️ [**UMAP**](2.3%20-%20unsupervised.md#232-dimensionality-reduction-pca-t-sne-umap)

## 1.6 HARDWARE & LATENCY CONSTRAINTS MATRIX
Il check finale prima di avviare l'addestramento.

| CONSTRAINTS PROGETTUALI | AZIONE / MODELLO CONSIGLIATO | MODELLI DA EVITARE (NO-GO) |
| :--- | :--- | :--- |
| **Latenza Inferenza Ultra-Low (Millisecondi)** | Linear/Logistic, Decision Trees, YOLO, TF-IDF, 1D-CNN. | KNN (O(N) in predizione), RF (>500 alberi), Mask R-CNN, LLMs. |
| **Deploy Edge Serverless (No GPU disponibile)** | Random Forest, XGBoost (inferenza CPU), SVM, Prophet. | Transformer, ViT, Deep CNNs, LSTM, Generative Autoencoders. |
| **VRAM GPU Molto Limitata (< 8 GB)** | 1D-CNN, GRU, ResNet-50 (batch < 16), YOLO Nano, LoRA su LLM. | Mask R-CNN, ViT, LLM Full-Parameter, U-Net 3D ad alta risoluzione. |
| **Cold Start / Small Data (Pochissimi Dati)** | SVM, Decision Trees, Naive Bayes. *(Per Visione/Testo: Transfer Learning obbligatorio).* | MLP tabulari, ViT addestrati da zero, LSTM, Generative GANs/VAE. |