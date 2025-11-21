# 00_load_packages.R
# -------------------
# Run this file first (examplatory packages and the way how to load them in different files)
# Loads (and installs if needed) all packages used in the ML project.

packages <- c(
  # Data manipulation
  "tidyverse",
  "data.table",
  "janitor",
  
  # Visualization
  "ggplot2",
  "GGally",
  "patchwork",
  "scales",
  
  # Modeling
  "mgcv",        # GAM
  "e1071",       # SVM
  "nnet",        # Neural Networks (simple)
  "keras",       # Neural Networks (advanced, optional)
  "caret",       # ML workflows + CV
  "rsample",     # Train-test splits
  "yardstick",   # Model evaluation
  
  # Preprocessing
  "recipes",
  
  # Model saving
  "rio",         # export/import
  "readr",
  
  # RMarkdown support (ensures reproducibility)
  "knitr",
  "rmarkdown"
)

# Install missing packages
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load packages
lapply(packages, library, character.only = TRUE)

message("All packages loaded successfully.")

