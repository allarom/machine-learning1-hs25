# 04_model_linear.R

# load packages
source(here::here("src", "00_load_packages.R"))

# -------------------------------------------------------------------
# 1. Load feature data
# -------------------------------------------------------------------
input_path <- here::here("data", "processed", "used_cars_features.csv")

message("Using feature data file: ", input_path)
if (!file.exists(input_path)) {
  stop("Feature data file not found at: ", input_path)
}

cars <- readr::read_delim(input_path, delim = ";", show_col_types = FALSE)

# -------------------------------------------------------------------
# 2. Prepare data for modelling
# -------------------------------------------------------------------
# keep only variables we actually use and drop rows with missing values
model_data <- cars |>
  dplyr::select(
    price_dollar,
    log_price,
    age,
    milage_k,
    accident_bin,
    brand,
    fuel_type,
    transmission,
    ext_col,
    int_col
  ) |>
  tidyr::drop_na()

# -------------------------------------------------------------------
# 3. Fit linear regression on log(price)
# -------------------------------------------------------------------
lm_linear <- stats::lm(
  log_price ~ age + milage_k + accident_bin +
    brand + fuel_type + transmission + ext_col + int_col,
  data = model_data
)

lm_summary <- summary(lm_linear)

# basic performance metrics
rmse_log <- sqrt(mean(lm_summary$residuals^2))
r2_linear <- lm_summary$r.squared

message("Linear model fitted with ", length(coef(lm_linear)), " coefficients.")
message("RMSE (log-price scale): ", round(rmse_log, 3))
message("R² (log-price scale): ", round(r2_linear, 3))

# -------------------------------------------------------------------
# 4. Add predictions back to data (useful for plots / diagnostics)
# -------------------------------------------------------------------
model_data <- model_data |>
  dplyr::mutate(
    pred_log_price    = stats::predict(lm_linear),
    resid_log_price   = log_price - pred_log_price,
    pred_price_dollar = exp(pred_log_price)
  )

# keep model and model_data in the environment for the report
assign("lm_linear", lm_linear, envir = .GlobalEnv)
assign("lm_linear_data", model_data, envir = .GlobalEnv)


