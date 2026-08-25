# EMIPredict AI: Financial Risk Assessment Platform

An end-to-end machine learning project that predicts EMI eligibility and estimates the maximum safe monthly EMI for an applicant. The project combines financial data preparation, feature engineering, model comparison, MLflow experiment tracking, and a deployable Streamlit dashboard.

## Project Highlights

- **Classification:** Categorises applicants as `Eligible`, `High_Risk`, or `Not_Eligible`.
- **Regression:** Estimates `max_monthly_emi`, the recommended affordable monthly EMI.
- **Data scale:** Analyses 400,000 financial profiles across five EMI scenarios.
- **Best models:** XGBoost for classification and regression.
- **Test performance:** 97.17% classification accuracy and 878.47 regression RMSE.

## Repository Structure

```text
.
├── app.py                         # Streamlit application entry point
├── artifacts/                     # Exported preprocessor and trained models
├── data/
│   ├── dashboard_sample.csv        # Lightweight sample used by Streamlit
│   └── emi_prediction_dataset.csv  # Full training dataset (optional in Git)
├── notebooks/EMIPredict.ipynb      # Fully executed ML notebook
├── reports/                        # Project report and presentation-ready charts
├── results/                        # Model comparison tables and execution report
├── requirements.txt                # Dashboard dependencies
└── requirements-notebook.txt       # Notebook and ML experiment dependencies
```

## Results

| Task | Selected model | Test result |
|---|---|---|
| EMI eligibility classification | XGBoost | Accuracy: 97.17%; weighted F1-score: 96.91%; ROC-AUC: 0.9981 |
| Maximum safe EMI regression | XGBoost | RMSE:  878.47; MAE:  415.67; R-squared: 0.9871 |

## Run the Dashboard Locally

### Prerequisites

- Python 3.11 recommended
- `pip`

### Installation

```bash
git clone https://github.com/<your-github-username>/EMIPredict-AI-Financial-Risk-Assessment.git
cd EMIPredict-AI-Financial-Risk-Assessment
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The application opens at `http://localhost:8501`.

## Run the Notebook

The full notebook requires the original dataset at `data/emi_prediction_dataset.csv`.

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/EMIPredict.ipynb
```

Run all cells from the notebook directory so its relative data and artifact paths resolve correctly.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository named `EMIPredict-AI-Financial-Risk-Assessment`.
2. Push this project folder to the repository.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Click **Deploy**. Streamlit installs the packages in `requirements.txt` automatically.

The dashboard only uses `data/dashboard_sample.csv`, trained model artifacts, and result files. It does **not** require the 400,000-row training dataset to run on Streamlit Cloud.

## Live Demo: [Try EMIPredit Live](https://gmpe5n3zcrolt3i6dlqfzx.streamlit.app/)

## Dataset Guidance

The original CSV is approximately 75 MB, which is below GitHub's 100 MB single-file limit. You can push it if the data is permitted to be shared and you need a fully reproducible notebook repository. However, the recommended approach is to exclude it from normal Git commits because it makes the repository large and slows cloning.

For deployment, keep `dashboard_sample.csv` in the repository and retain the original training dataset locally, in a cloud drive, a GitHub Release, or Git LFS. Do not upload the original dataset if it contains confidential or personally identifiable information.

## Important Files

- `reports/EMIPredict_Project_Report.docx`: Professional project report.
- `reports/graphs/`: Charts used in the report.
- `reports/graphs/eda/`: Complete EDA visual set, including distributions, bivariate analysis, scatter plot, correlation heatmap, and pair plots.
- `results/EMIPredict_execution_report.html`: Browser-readable executed notebook output.
- `notebooks/EMIPredict.ipynb`: Full ML workflow and MLflow logging implementation.

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, MLflow, Matplotlib, Seaborn, Joblib, and Streamlit.
