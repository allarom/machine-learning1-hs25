# 04_model_linear.R

# load packages and helpers
source(here::here("src", "00_load_packages.R"))
source(here::here("src", "helpers.R"))

# -------------------------------------------------------------------
# 1. Load feature data
# -------------------------------------------------------------------
input_path <- here::here("data", "processed", "used_cars_features.csv")
linear_dir <- here::here("report", "models", "linear")

message("Using feature data file: ", input_path)
if (!file.exists(input_path)) {
  stop("Feature data file not found at: ", input_path)
}
dir.create(linear_dir, recursive = TRUE, showWarnings = FALSE)

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
# 3. Train/test split (shared across models for fair comparison)
# -------------------------------------------------------------------
split <- get_train_test_split(nrow(model_data), prop = 0.8, seed = 42)
train_data <- model_data[split$train_idx, ]
test_data  <- model_data[split$test_idx, ]

# -------------------------------------------------------------------
# 4. Fit linear regression on log(price)
# -------------------------------------------------------------------
lm_linear <- stats::lm(
  log_price ~ age + milage_k + accident_bin +
    brand + fuel_type + transmission + ext_col + int_col,
  data = train_data
)

# Predictions (train/test)
pred_train <- stats::predict(lm_linear, newdata = train_data)
pred_test  <- stats::predict(lm_linear, newdata = test_data)

# RMSE on log scale (comparable to SVM/NN if they also predict log_price)
rmse_train_log <- sqrt(mean((train_data$log_price - pred_train)^2))
rmse_test_log  <- sqrt(mean((test_data$log_price  - pred_test)^2))
mae_train_log <- mean(abs(train_data$log_price - pred_train))
mae_test_log  <- mean(abs(test_data$log_price  - pred_test))

# R² on train (from model) + R² on test (manual)
r2_train <- summary(lm_linear)$r.squared
r2_test  <- 1 - sum((test_data$log_price - pred_test)^2) /
  sum((test_data$log_price - mean(test_data$log_price))^2)

message("Linear model fitted with ", length(coef(lm_linear)), " coefficients.")
message("RMSE train (log): ", round(rmse_train_log, 3))
message("RMSE test  (log): ", round(rmse_test_log, 3))
message("MAE  train (log): ", round(mae_train_log, 3))
message("MAE  test  (log): ", round(mae_test_log, 3))
message("R² train (log): ", round(r2_train, 3))
message("R² test  (log): ", round(r2_test, 3))

# -------------------------------------------------------------------
# 5. Add predictions back to full data (useful for plots / diagnostics)
# -------------------------------------------------------------------
model_data <- model_data |>
  dplyr::mutate(
    split           = ifelse(dplyr::row_number() %in% split$train_idx, "train", "test"),
    pred_log_price   = stats::predict(lm_linear, newdata = model_data),
    resid_log_price  = log_price - pred_log_price,
    pred_price_dollar = exp(pred_log_price)
  )

# Metrics table (for report + later final comparison table)
lm_metrics <- tibble::tibble(
  model = "Linear regression",
  target = "log_price",
  rmse_train = rmse_train_log,
  rmse_test = rmse_test_log,
  mae_train = mae_train_log,
  mae_test = mae_test_log,
  r2_train = r2_train,
  r2_test = r2_test
)

# Persist metrics for comparisons
readr::write_csv(lm_metrics, file.path(linear_dir, "linear_log_price_metrics.csv"))

# keep objects in the environment for the report
assign("lm_linear", lm_linear, envir = .GlobalEnv)
assign("lm_linear_data", model_data, envir = .GlobalEnv)
assign("lm_metrics", lm_metrics, envir = .GlobalEnv)
assign("lm_linear_rmse_log", rmse_test_log, envir = .GlobalEnv)
assign("lm_linear_r2_log", r2_test, envir = .GlobalEnv)
