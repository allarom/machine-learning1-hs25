# machine-learning1-hs25

Group project for **Applied Machine Learning and Predictive Modelling 1** using the Kaggle dataset **Used Car Price Prediction**.  

To install libraries you need to run file 00_load_packages.R.
To compile markdown files in R-Studio: in RMD files you have a button called "Knit", there you can specify the format.

In my setup (Mac, Visual Code Studio) I needed to specify Pandoc location in .Renviron file (placed in my root folder ~, so path was ~/.Renviron)

There I added:

```RSTUDIO_PANDOC=/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/x86_64```


You can complile analysis.html file by running this command in the R:

```source("src/00_load_packages.R")
compile_analysis_report()```

Or on Mac by running ```Cmd+Shift+K```, K stands for "knitting"


---

## Repository structure (overview)

- `data/raw/` – original dataset (e.g., `used_cars.csv`)
- `data/processed/` – cleaned + engineered datasets (e.g., `used_cars_features.csv`) and shared split (`train_test_split.rds`)
- `src/` – pipeline scripts (00…10)
  - `00_load_packages.R` – installs/loads packages + provides `compile_analysis_report()`
  - `01_data_cleaning.R` – raw → cleaned
  - `02_feature_engineering.R` – cleaned → features
  - `03_exploratory_analysis.R` – EDA + plots
  - `04_model_linear.R` … `09_model_nn.R` – model scripts
  - `10_model_comparison.R` – consolidated comparison
  - `helpers.R` – shared helpers/utilities
- `report/`
  - `analysis.Rmd` – main report source
  - `analysis.html` – rendered output
  - `styles.css` – report styling
  - `plots/`, `models/` – generated artifacts (figures + metrics/predictions)

---

