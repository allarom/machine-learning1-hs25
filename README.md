# machine-learning1-hs25

Group project for **Applied Machine Learning and Predictive Modelling 1** using the Kaggle dataset **Used Car Price Prediction**.  

## Repository structure

- `data/raw/` – original dataset (e.g., `used_cars.csv`)
- `data/processed/` – cleaned + engineered datasets (e.g., `used_cars_features.csv`) and shared split (`train_test_split.rds`)
- `src/` – pipeline scripts (00…10)
  - `00_load_packages.R` – installs/loads packages + provides `compile_analysis_report()`
  - `01_data_cleaning.R` – raw → cleaned
  - `02_feature_engineering.R` – cleaned → features
  - `03_exploratory_analysis.R` – EDA + plots
  - `04_model_linear.R` – Linear Regression (Julio)
  - `05_model_glm_binomial.R` – GLM Binomial (Tashi)
  - `06_model_glm_poisson.R` –GLM Poisson (Julio)
  - `07_model_gam.R` – GAM (Tashi)
  - `08_model_svm.R` – SVM (Alla)
  - `09_model_nn.R` – Neural Network (Alla)
  - `10_model_comparison.R` – consolidated comparison
  - `helpers.R` – shared helpers/utilities (incl. shared train/test split usage)
- `report/`
  - `analysis.Rmd` – main report source
  - `analysis.html` – rendered output
  - `styles.css` – report styling
  - `plots/`, `models/` – generated artifacts (figures + metrics/predictions)