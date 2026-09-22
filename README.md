# SECOM Wafer Yield Optimization & In-Line Process Defect Isolation

> **Third Year (T.E.) Electronics & Computer Engineering Mini Project**  
> **Subject Area:** Microelectronics, Integrated Circuit (IC) Manufacturing, & Machine Learning Quality Control

---

## 📌 Problem Statement

In modern 300mm silicon wafer fabrication facilities, sub-micron particle contamination and tool parameter drift during photolithography, plasma etching, and Chemical Mechanical Planarization (CMP) directly cause wafer defects. Detecting defect patterns early in the manufacturing line prevents unviable wafers from progressing through expensive packaging, wire bonding, and final testing stages.

This project analyzes sensor telemetry from **1,500 silicon wafer production runs** across 15 process parameters operating at a 14nm FinFET process node to predict wafer yield status (`Pass` vs `Defect`) and isolate critical sensor control limits.

---

## 🔬 Microelectronics & Process Engineering Context

During integrated circuit fabrication, maintaining Statistical Process Control (SPC) within Lower Control Limits (LCL) and Upper Control Limits (UCL) is critical for high fabrication yield:

1. **Plasma Etch RF Reflected Power (>465 W):** Elevated reflected power indicates plasma instability inside the etch chamber, leading to dielectric oxide breakdown and increased interconnect leakage current (`>0.075 μA`).
2. **CMP Polishing Pad Wear (>55 μm):** CMP pad degradation disrupts slurry distribution across the wafer surface, causing oxide thickness non-uniformity and micro-scratches.

---

## ⚙️ Methodology & Machine Learning Pipeline

1. **Data Preprocessing & Imputation:** Handle 2.5% real-world sensor dropout (missing telemetry values) using median imputation (`SimpleImputer`).
2. **Handling Class Imbalance:** The SECOM dataset exhibits severe class imbalance (94.2% Pass vs 5.8% Defect). Synthetic Minority Over-sampling Technique (`SMOTE`) is applied on the training split to balance class distribution.
3. **Classification Model:** Random Forest Classifier (`n_estimators=100`, `max_depth=6`) trained on SMOTE-resampled sensor data.
4. **Model Evaluation & Threshold Tuning:** Standard `predict_proba()` outputs are evaluated across decision thresholds to balance under-detection against over-rejection.

---

## 📊 Model Performance Analysis & Technical Trade-offs

In semiconductor quality control, classification errors carry asymmetric technical consequences:

* **Under-detection (False Negative):** A defective wafer goes unnoticed and undergoes full packaging and testing before failing final probe tests.
* **Over-rejection (False Positive):** A good wafer is flagged for secondary manual surface inspection.

| Decision Threshold | Recall (Defect Catch Rate) | Precision | False Negatives (Under-detected) | False Positives (Over-rejected) | Technical Impact |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50 (Standard)** | 72.73% | 80.00% | 6 Wafers | 4 Wafers | Higher risk of passing defective silicon |
| **0.35 (Optimized)** | **90.91%** | **66.67%** | **2 Wafers** | **10 Wafers** | Maximizes defect capture prior to packaging |

---

## 🛠️ Statistical Process Control Limits (LCL / UCL)

| Sensor Parameter | Nominal Target | LCL | UCL | Corrective Action Protocol |
| :--- | :---: | :---: | :---: | :--- |
| **RF Reflected Power** | 450.0 W | 430.0 W | **465.0 W** | Recalibrate match network & RF generator |
| **CMP Pad Wear** | 30.0 μm | 5.0 μm | **55.0 μm** | Replace CMP diamond conditioner pad |
| **Deposition Temp** | 350.0 °C | **335.0 °C** | 370.0 °C | Inspect thermal couple heating element |
| **Leakage Current** | 0.050 μA | 0.010 μA | **0.075 μA** | Perform in-situ chamber purge test |

---

## 📂 Repository Layout

```text
secom-yield-prediction/
├── app.py                          # Streamlit Interactive Monitoring Dashboard
├── requirements.txt                 # Dependencies manifest
├── README.md                        # Mini project report & analysis
├── Project_Report.pdf               # Academic Mini Project PDF Report
└── secom_wafer_data-v4.csv         # Sensor telemetry dataset (1,500 wafer records)
```

---

## 🚀 Setup & Execution

1. **Clone Repository:**
   ```bash
   git clone https://github.com/your-username/secom-yield-prediction.git
   cd secom-yield-prediction
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
