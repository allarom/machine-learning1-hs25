# 01_data_cleaning.R
source("src/00_load_packages.R")


# 1) Load data -------------------------------------------------------------
raw_path <- "data/raw/used_cars.csv"
cars_raw <- readr::read_csv(raw_path, show_col_types = FALSE)

# Quick look
glimpse(cars_raw)
summary(cars_raw)
