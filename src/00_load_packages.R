# 00_load_packages.R
# -------------------
# Run this file first (examplatory packages and the way how to load them in different files)
# Loads (and installs if needed) all packages used in the ML project.

packages <- c(
  # Data manipulation
  "tidyverse",
  "data.table",
  "janitor",
  "here",
  
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

# Compile the existing report/analysis.Rmd from any working directory.
# Example: after sourcing this file, run compile_analysis_report().
compile_analysis_report <- function(
  input = here("report", "analysis.Rmd"),
  output_dir = here("report"),
  output_format = "html_document"
) {
  if (!file.exists(input)) {
    stop("Cannot find analysis file at: ", input)
  }
  rmarkdown::render(
    input = input,
    output_dir = output_dir,
    output_format = output_format,
    envir = new.env()
  )
}
