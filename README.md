# 🛡️ Insurance Claim Intelligence

An interactive **Machine Learning + Data Analytics** application for exploring insurance claims and estimating patterns associated with reported fraud. Built with Python, Pandas, Scikit-learn, and Streamlit.

> **Responsible-use notice:** This is an educational portfolio project, not a validated insurance fraud investigation system. A model prediction is not proof of fraud and must never be used as the sole basis to approve, deny, delay, or investigate a real claim. Real-world decisions require verified evidence, human review, independent validation, privacy safeguards, and compliance with applicable law.

## 🚀 Live Demo

**Streamlit App:** _Add your deployed Streamlit URL here after deployment. (Streamlit Community Cloud will provide this URL.)_

## ✨ Features

- **Interactive overview:** total claims, reported-fraud share, and dataset health indicators.
- **Claim explorer:** filter by incident type, incident severity, and target label.
- **Data visualizations:** class distribution, claim amount summaries, incident patterns, and claim amount boxplots.
- **Interactive prediction form:** enter policy and incident details to obtain a model-estimated result.
- **Model comparison:** Random Forest and Gradient Boosting evaluated on a held-out test set.
- **Performance reporting:** Accuracy, Precision, Recall, F1 score, classification report, and confusion matrix.
- **Download filtered records:** export the currently filtered dataset as CSV.
- **Cloud-ready structure:** trained model and metrics are stored separately from application code.

## 🖼️ Application Screenshots

After running and deploying the app, capture screenshots and upload them to the `screenshots/` folder using these filenames.

| Overview | Claim Explorer |
|---|---|
| `screenshots/home.png` | `screenshots/eda.png` |

| Claim Prediction | Model Performance |
|---|---|
| `screenshots/prediction.png` | `screenshots/model_performance.png` |

To display the images here after uploading them, add:

```markdown
![Overview](screenshots/home.png)
![Claim Explorer](screenshots/eda.png)
![Claim Prediction](screenshots/prediction.png)
![Model Performance](screenshots/model_performance.png)
```

## 🧰 Tech Stack

- **Language:** Python
- **App:** Streamlit
- **Data handling:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visual analytics:** Matplotlib, Seaborn
- **Model persistence:** Joblib
- **Dataset format:** Excel (`.xlsx`)

## 📊 Dataset

The included dataset contains **1,000 rows and 39 columns**. The target column is `fraud_reported`:

- `Y` — fraud reported in the dataset
- `N` — no fraud reported in the dataset

The dataset contains policy, insured-customer, incident, vehicle, and claim information. The target describes the recorded label in this dataset; it should not be interpreted as a definitive real-world finding.

## ⚙️ Machine Learning Workflow

1. Load the Excel dataset and replace placeholder missing values (`?`).
2. Separate the target `fraud_reported` from input features.
3. Remove direct identifiers and free-text location fields from the model features.
4. Split the data into training and test sets using stratification.
5. Impute missing numeric and categorical values within a scikit-learn pipeline.
6. One-hot encode categorical features, handling unseen categories safely.
7. Train and compare Random Forest and Gradient Boosting classifiers.
8. Compare test-set Accuracy, Precision, Recall, and F1 score.
9. Select the model with the highest test-set F1 score and save the model pipeline and evaluation metrics.

## 📈 Evaluation Metrics

- **Accuracy:** proportion of all test examples predicted correctly.
- **Precision:** proportion of predicted fraud cases that are labelled fraud in the test data.
- **Recall:** proportion of labelled fraud cases identified by the model.
- **F1 score:** harmonic mean of Precision and Recall.

Because fraud cases are the minority class in this dataset, accuracy alone is not enough to assess usefulness. Review Precision, Recall, F1 score, and the confusion matrix together.

## 📁 Project Structure

```text
INSURANCE_CLAIM_PREDICTION/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── insurance_claims.xlsx
└── models/
    ├── insurance_fraud_model.joblib
    └── metrics.json
```

## 💻 Run Locally

### 1. Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username:

```bash
git clone https://github.com/priyadharshini2002-source/insurance-claim-prediction.git
cd insurance-claim-prediction
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This generates:

- `models/insurance_fraud_model.joblib`
- `models/metrics.json`

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal.

## ☁️ Deploy on Streamlit Community Cloud

1. Create a GitHub repository named `insurance-claim-prediction`.
2. Upload the project files and commit them.
3. Confirm that `app.py`, `requirements.txt`, `data/insurance_claims.xlsx`, `models/insurance_fraud_model.joblib`, and `models/metrics.json` are in the repository.
4. Open Streamlit Community Cloud and choose **Create app**.
5. Select your repository, branch, and `app.py` as the main file.
6. Deploy, then add the resulting public URL to the Live Demo section above.

If you change the training code, rerun `python train_model.py` and commit the newly generated model and metrics.

## 🔮 Future Enhancements

- Add cross-validation and threshold tuning for the minority class.
- Add calibration analysis and precision-recall curves.
- Add explainability tools to help reviewers understand influential features.
- Add data drift monitoring and an independent external validation set.
- Add audit logging, access controls, and privacy protections before any real-world pilot.

## ⚠️ Limitations

- Results depend on the quality, size, and representativeness of the provided dataset.
- Test-set performance does not guarantee future or real-world performance.
- Categorical patterns can change between datasets and over time.
- False positives and false negatives are possible.
- A high-risk estimate is not proof of wrongdoing.

## 👩‍💻 Author

**Priyadharshini S.**  
MSc Data Science | Machine Learning | Data Analytics

## 📄 License

No license is included by default. Add a `LICENSE` file if you want to grant others explicit permission to reuse, modify, or distribute the code.
