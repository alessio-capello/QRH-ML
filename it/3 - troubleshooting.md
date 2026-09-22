# 3. TROUBLESHOOTING CHECKLISTS

**Quick links**
* [0. Come usare questo QRH](0%20-%20hot%20to%20use.md#1-decision-matrix)
* [2.1 Baseline ML & Metodi Lineari](2.1%20-%20baseline_lineari.md)
* [2.3 Unsupervised & Rappresentazione](2.3%20-%20unsupervised.md)
* [2.7 NLP](2.7%20-%20NLP.md)
* [2.8 Reinforcement Learning](2.8%20-%20reinforcement%20learning.md)
* Sezioni: [Loss = NaN](#-sintomo-loss--nan-not-a-number), [Flat Loss](#-sintomo-loss-piatta-fin-dalla-prima-epoca-il-modello-non-impara), [Underfitting](#-sintomo-alta-loss-in-training-alta-loss-in-validation-underfitting-grave), [Overfitting](#-sintomo-bassa-loss-in-training-validation-loss-diverge-o-sale-overfitting-cronico), [OOM](#-sintomo-oom-out-of-memory-error-sulla-gpu)

Questa sezione contiene le procedure di emergenza per la diagnostica e la risoluzione dei colli di bottiglia più comuni durante il ciclo di vita di un modello (Training, Data Pipeline, Produzione).

## 3.1 TRAINING FAILURES (LOSS & GRADIENTI)
Procedure per quando l'addestramento fallisce matematicamente o si blocca.

### [SINTOMO] Loss = NaN (Not a Number)
* **Diagnosi 1: Gradient Explosion.** I gradienti sono cresciuti in modo esponenziale, superando il limite dei float32. Tipico in RNN, LSTM e Transformer senza warmup.
  * *Quick Fix:* Inserire Gradient Clipping (`clipnorm=1.0` o `clipvalue=0.5`). Abbassare drasticamente il `learning_rate`.
* **Diagnosi 2: Input Corrotti (Division by Zero / Log of Zero).**
  * *Quick Fix:* Controllare il dataset per valori `NaN` o `Inf` passati inosservati. Aggiungere un epsilon (`+ 1e-8`) prima di operazioni logaritmiche o divisioni nel calcolo della loss o nelle funzioni custom.
* **Diagnosi 3: Learning Rate Eccessivo.**
  * *Quick Fix:* Ridurre il LR di un ordine di grandezza (es. da 1e-3 a 1e-4). 

### [SINTOMO] Loss Piatta fin dalla prima epoca (Il modello non impara)
* **Diagnosi 1: Vanishing Gradients.** Il segnale di errore si azzera propagandosi all'indietro (tipico di reti profonde con funzioni Sigmoide/Tanh).
  * *Quick Fix:* Sostituire le attivazioni con `ReLU`, `LeakyReLU` o `Mish`. Implementare la normalizzazione (`Batch Normalization` o `Layer Normalization`). Controllare l'inizializzazione dei pesi (usare `He / Glorot`).
* **Diagnosi 2: Input Scale Discrepancy.** I dati in ingresso hanno magnitudo totalmente diverse (es. una feature varia da 0 a 1, un'altra da 1.000 a 10.000).
  * *Quick Fix:* Arrestare il training. Applicare `StandardScaler` o `MinMaxScaler` su tutte le feature numeriche.

---

## 3.2 PERFORMANCE ISSUES (UNDERFITTING & OVERFITTING)
Procedure per quando il modello completa l'addestramento ma le metriche sono errate.

### [SINTOMO] Alta Loss in Training, Alta Loss in Validation (Underfitting Grave)
* **Diagnosi:** Il modello non ha la capacità (i parametri) per mappare la complessità del problema, oppure è eccessivamente regolarizzato.
  * *Quick Fix 1 (Capacità):* Aumentare la complessità (più alberi in RF, più layer/neuroni in MLP, passare da ResNet-18 a ResNet-50).
  * *Quick Fix 2 (Regolarizzazione):* Ridurre il Dropout, abbassare le penalità L1/L2 (Weight Decay), diminuire `min_samples_leaf` negli alberi.
  * *Quick Fix 3 (Feature):* Fare Feature Engineering (creare interazioni polinomiali).

### [SINTOMO] Bassa Loss in Training, Validation Loss diverge o sale (Overfitting Cronico)
* **Diagnosi:** Il modello sta memorizzando a memoria i dati di training (rumore compreso) e non generalizza sui dati nuovi.
  * *Quick Fix 1 (Early Stopping):* Fermare il training al punto di flesso prima che la validation loss inizi a salire.
  * *Quick Fix 2 (Data):* Aumentare il dataset o usare massicciamente Data Augmentation (immagini/audio).
  * *Quick Fix 3 (Regolarizzazione):* Aumentare il Dropout (es. 0.5), aggiungere Weight Decay (L2), ridurre la profondità del modello (`max_depth` per alberi, ridurre i layer per reti neurali).

---

## 3.3 DATA PIPELINE & HARDWARE FAILURES
Procedure per problemi di infrastruttura o contaminazione logica dei dati.

### [SINTOMO] OOM (Out Of Memory) Error sulla GPU
* **Diagnosi:** Il tensore dei gradienti e le feature map superano la VRAM fisica (tipico nei Transformer o nella Computer Vision ad alta risoluzione).
  * *Quick Fix 1 (Immediata):* Dimezzare iterativamente il `batch_size` (es. 64 -> 32 -> 16).
  * *Quick Fix 2 (Gradient Accumulation):* Se un `batch_size` basso rende i gradienti instabili, simulare batch grandi calcolando i gradienti su batch piccoli e aggiornando i pesi solo ogni N step.
  * *Quick Fix 3 (Precisione):* Attivare Mixed Precision Training (FP16 / BF16). Dimezza il consumo di VRAM senza quasi perdere accuratezza.

### [SINTOMO] Modello "Perfetto" in Test (es. Accuracy 99%) ma disastroso nel mondo reale
* **Diagnosi: Data Leakage.** Informazioni sul target sono accidentalmente incluse nelle feature, o il Validation Set è contaminato dal Training Set.
  * *Quick Fix 1 (Time Series):* Controllare di aver fatto uno split temporale stretto. Mai usare scissioni puramente casuali (`train_test_split`) sulle serie storiche.
  * *Quick Fix 2 (Scaling):* Controllare che `StandardScaler.fit()` sia stato applicato SOLO sul set di Training, e non sull'intero dataset primario.
  * *Quick Fix 3 (Feature Implicite):* Assicurarsi che nessuna feature sia una conseguenza diretta della variabile target (es. usare i "giorni di ricovero" per prevedere se "il paziente verrà ricoverato").

### [SINTOMO] Accuracy altissima (99%), ma Precision/Recall su una classe sono a zero
* **Diagnosi: Class Imbalance Severo.** (Es. Il 99% delle transazioni è legittima, l'1% è frode. Prevedere sempre "Legittimo" regala il 99% di accuracy a costo zero).
  * *Quick Fix 1 (Metriche):* Smettere di guardare l'Accuracy. Usare F1-Score, Precision, Recall e AUC-PR.
  * *Quick Fix 2 (Algoritmico):* Usare funzioni di Loss pesate (Focal Loss, Class Weights nell'algoritmo).
  * *Quick Fix 3 (Dati):* Sovra-campionare la classe minoritaria (SMOTE) o Sotto-campionare la maggioritaria (solo nel set di training).

---

## 3.4 INFERENCE & PRODUCTION
Procedure per il monitoraggio e le performance in fase di deployment.

### [SINTOMO] Latenza d'Inferenza troppo alta (Timeout o ritardi API)
* **Diagnosi:** Il modello salvato (in formato raw, es. PyTorch `.pt` o `.pkl`) esegue calcoli non necessari (es. grafi di gradienti) o non è ottimizzato per l'hardware target.
  * *Quick Fix 1:* Esportare il modello nei formati compilati ONNX o TensorRT per accelerazione hardware.
  * *Quick Fix 2:* Quantizzazione post-training. Passare i pesi del modello da Float32 a INT8 (velocità quadruplicata, perdita di precisione minima).
  * *Quick Fix 3:* Raggruppare le richieste API concorrenti in micro-batch (Dynamic Batching) invece di processarle una alla volta.

### [SINTOMO] Prestazioni in decadimento progressivo dopo N mesi dal Deploy
* **Diagnosi: Concept / Data Drift.** Le abitudini degli utenti (Concept) o i formati fisici dei sensori in ingresso (Data) sono mutati rispetto ai dati usati nel training storico.
  * *Quick Fix (Monitoraggio):* Implementare test statistici (es. Kolmogorov-Smirnov Test, PSI - Population Stability Index) sulla distribuzione dell'input in produzione.
  * *Quick Fix (Risoluzione):* Pianificare un retraining schedulato o passare a un paradigma di Continual Learning includendo campioni di dati freschi degli ultimi 30 giorni (dando loro maggior peso algoritmico).