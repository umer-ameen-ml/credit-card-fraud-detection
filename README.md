# 🔍 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a **Decision Tree Classifier** with **SMOTE** (Synthetic Minority Oversampling Technique) to handle class imbalance.

---

## 📌 Project Overview

Credit card fraud detection is a classic imbalanced classification problem — fraudulent transactions are very rare compared to legitimate ones. This project tackles that challenge by:

- Training a Decision Tree model **before** and **after** applying SMOTE
- Comparing performance to show the impact of oversampling
- Visualizing class distributions, confusion matrix, and feature importance

---

## 📁 Project Structure

```
fraud-detection/
│
├── fraud_detection.py          # Main script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files to exclude from git
├── README.md                   # Project documentation
│
└── outputs/                    # Generated after running the script
    ├── before_smote_distribution.png
    ├── after_smote_distribution.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 📊 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- **284,807** transactions
- **492** fraudulent (Class = 1)
- **284,315** legitimate (Class = 0)
- Features are PCA-transformed (V1–V28) + `Time` and `Amount`

> ⚠️ The dataset is **not included** in this repo due to its large size. Download it from Kaggle and place `creditcard.csv` in your local path.

---

## ⚙️ How It Works

1. **Load Data** — Read the CSV and split features/target
2. **Train/Test Split** — 80/20 stratified split
3. **Baseline Model** — Decision Tree trained on imbalanced data
4. **SMOTE** — Oversample the minority (fraud) class
5. **SMOTE Model** — Decision Tree trained on balanced data
6. **Evaluation** — Accuracy, classification report, confusion matrix
7. **Visualization** — 4 charts saved as PNG files

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Before SMOTE | ~99% (biased toward majority class) |
| After SMOTE | Improved fraud recall & F1-score |

> The model after SMOTE performs better at **catching actual fraud** (higher recall for Class 1).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Download `creditcard.csv` from Kaggle and update the path in `fraud_detection.py`:

```python
df = pd.read_csv(r"path\to\your\creditcard.csv")
```

### 4. Run the Script

```bash
python fraud_detection.py
```

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Model training and evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `matplotlib` | Data visualization |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Umer Ameen**
