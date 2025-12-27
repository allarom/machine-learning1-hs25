# 06_model_glm_poisson.R

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
#    Poisson needs a non-negative, count-like response.
#    We use mileage in thousands as an event-like count proxy.
# -------------------------------------------------------------------
model_data <- dplyr::transmute(
  cars,
  milage_k_count = pmax(0L, as.integer(round(milage_k))),
  age = age,
  accident_bin = accident_bin,
  brand = brand,
  fuel_type = fuel_type,
  transmission = transmission,
  ext_col = ext_col,
  int_col = int_col
)

model_data <- tidyr::drop_na(model_data)

# -------------------------------------------------------------------
# 3. Train/test split (same idea as other models)
# -------------------------------------------------------------------
set.seed(123)
n <- nrow(model_data)
train_idx <- sample.int(n, size = floor(0.8 * n))

train_data <- model_data[train_idx, ]
test_data  <- model_data[-train_idx, ]

# -------------------------------------------------------------------
# 4. Fit Poisson GLM (log link)
# -------------------------------------------------------------------
glm_poisson <- stats::glm(
  milage_k_count ~ age + accident_bin +
    brand + fuel_type + transmission + ext_col + int_col,
  data = train_data,
  family = stats::poisson(link = "log")
)

# -------------------------------------------------------------------
# 5. Metrics / diagnostics
# -------------------------------------------------------------------
pred_train <- stats::predict(glm_poisson, newdata = train_data, type = "response")
pred_test  <- stats::predict(glm_poisson, newdata = test_data,  type = "response")

rmse_train <- sqrt(mean((train_data$milage_k_count - pred_train)^2))
rmse_test  <- sqrt(mean((test_data$milage_k_count  - pred_test)^2))

pseudo_r2 <- 1 - (glm_poisson$deviance / glm_poisson$null.deviance)
dispersion <- sum(stats::residuals(glm_poisson, type = "pearson")^2) / glm_poisson$df.residual
aic_val <- AIC(glm_poisson)

poisson_metrics <- tibble::tibble(
  model = "Poisson GLM",
  target = "milage_k_count",
  rmse_train = rmse_train,
  rmse_test = rmse_test,
  AIC = aic_val,
  pseudo_R2_deviance = pseudo_r2,
  dispersion_pearson = dispersion
)

message("Poisson GLM fitted.")
message("RMSE train (count): ", round(rmse_train, 3))
message("RMSE test  (count): ", round(rmse_test, 3))
message("Dispersion (Pearson): ", round(dispersion, 3))

# -------------------------------------------------------------------
# 6. Add predictions back to full data (for plots)
# -------------------------------------------------------------------
model_data <- model_data |>
  dplyr::mutate(
    split = ifelse(dplyr::row_number() %in% train_idx, "train", "test"),
    pred_milage_k = stats::predict(glm_poisson, newdata = model_data, type = "response"),
    resid_milage_k = milage_k_count - pred_milage_k
  )

# RMSE on response scale (count proxy)
rmse_poisson <- sqrt(mean(model_data$resid_milage_k^2))

assign("glm_poisson", glm_poisson, envir = .GlobalEnv)
assign("glm_poisson_data", model_data, envir = .GlobalEnv)
assign("poisson_metrics", poisson_metrics, envir = .GlobalEnv)