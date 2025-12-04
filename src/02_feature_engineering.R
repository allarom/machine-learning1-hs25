# 02_feature_engineering.R

source("src/00_load_packages.R")

# paths for cleaned input and feature output
input_path <- here::here("data", "processed", "used_cars_cleaned.csv")
output_path <- here::here("data", "processed", "used_cars_features.csv")

message("Using data file: ", input_path)
if (!file.exists(input_path)) {
  stop("Data file not found at: ", input_path)
}

cars <- readr::read_delim(input_path, delim = ";", show_col_types = FALSE)

# add features used across the models
cars <- cars |>
  dplyr::mutate(
    # car age in years
    age = 2025 - model_year,
    
    # mileage in thousands
    milage_k = round(milage / 1000),
    
    # Log price for better behaviour in linear models
    log_price = log(price_dollar),
    
    # Binary accident indicator (0 = no accident, 1 = at least one accident)
    accident_bin = ifelse(accident == "None reported", 0L, 1L),
    
    # categorical vars as factors
    brand        = factor(brand),
    fuel_type    = factor(fuel_type),
    transmission = factor(transmission),
    ext_col      = factor(ext_col),
    int_col      = factor(int_col)
  )

readr::write_delim(cars, output_path, delim = ";")

message("Wrote engineered data to: ", output_path)


