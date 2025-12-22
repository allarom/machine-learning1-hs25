# 09_model_nn.R
# Neural network regression for log_price using caret (nnet engine) and (optionally) neuralnet.
# Uses an 80/20 split, 3-fold cross validation for hyperparameter tuning,
# and saves fitted models, predictions, and metrics for comparison.

source("src/00_load_packages.R")

data_path <- here::here("data", "processed", "used_cars_features.csv")

if (!file.exists(data_path)) {
  stop("Data file not found at: ", data_path)
}

cars <- readr::read_delim(data_path, delim = ";", show_col_types = FALSE)

# Model-ready dataset: keep engineered predictors and convert categoricals
model_data <- cars |>
  dplyr::transmute(
    log_price = log_price,
    age = age,
    milage_k = milage_k,
    horsepower = horsepower,
    cylinders = cylinders,
    brand = factor(brand),
    fuel_type = factor(fuel_type),
    transmission = factor(transmission),
    ext_col = factor(ext_col),
    int_col = factor(int_col),
    accident_bin = factor(accident_bin)
  )

set.seed(42)
split <- rsample::initial_split(model_data, prop = 0.8)
train_data <- rsample::training(split)
test_data <- rsample::testing(split)

compute_metrics <- function(pred, truth) {
  results_tbl <- tibble::tibble(truth = truth, estimate = as.numeric(pred))
  metrics <- yardstick::metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq)
  metrics(results_tbl, truth = truth, estimate = estimate)
}

nn_dir <- here::here("report", "models", "nn")
dir.create(nn_dir, recursive = TRUE, showWarnings = FALSE)

# caret + nnet: recipe handles dummies and scaling; 3-fold cross validation for tuning
nn_recipe <- recipes::recipe(
  log_price ~ age + milage_k + horsepower + cylinders + brand + fuel_type + transmission + ext_col + int_col + accident_bin,
  data = train_data
) |>
  recipes::step_dummy(recipes::all_nominal_predictors()) |>
  recipes::step_zv(recipes::all_predictors()) |>
  recipes::step_center(recipes::all_numeric_predictors()) |>
  recipes::step_scale(recipes::all_numeric_predictors())

ctrl <- caret::trainControl(method = "cv", number = 3)
nn_grid <- expand.grid(
  size = c(3, 5),
  decay = c(0, 0.001)
)

nn_caret <- caret::train(
  nn_recipe,
  data = train_data,
  method = "nnet",
  linout = TRUE,
  trace = FALSE,
  maxit = 300,
  trControl = ctrl,
  tuneGrid = nn_grid
)

caret_pred <- stats::predict(nn_caret, newdata = test_data)
caret_metrics <- compute_metrics(caret_pred, test_data$log_price) |>
  dplyr::mutate(model = "caret_nnet")

saveRDS(nn_caret, file.path(nn_dir, "nn_log_price_caret.rds"))
readr::write_delim(
  dplyr::bind_cols(test_data, tibble::tibble(pred_log_price = as.numeric(caret_pred))),
  file.path(nn_dir, "nn_log_price_caret_predictions.csv"),
  delim = ";"
)

manual_metrics <- NULL

if (requireNamespace("neuralnet", quietly = TRUE)) {
  # neuralnet package: manual dummy variables and scaling from training set
  train_mm <- stats::model.matrix(log_price ~ ., data = train_data)
  test_mm <- stats::model.matrix(log_price ~ ., data = test_data)

  train_x <- train_mm[, -1, drop = FALSE] # drop intercept
  test_x <- test_mm[, -1, drop = FALSE]

  center <- apply(train_x, 2, mean)
  scale <- apply(train_x, 2, sd)
  scale[scale == 0] <- 1 # guard against zero variance

  train_xs <- scale(train_x, center = center, scale = scale)
  test_xs <- scale(test_x, center = center, scale = scale)

  # Ensure safe column names for formula parsing
  safe_names <- make.names(colnames(train_xs))
  colnames(train_xs) <- safe_names
  colnames(test_xs) <- safe_names

  nn_train_df <- as.data.frame(train_xs)
  nn_train_df$log_price <- train_data$log_price

  nn_test_df <- as.data.frame(test_xs)
  nn_test_df$log_price <- test_data$log_price

  # Sample to keep manual neuralnet runtime manageable
  set.seed(42)
  if (nrow(nn_train_df) > 1800) {
    nn_train_df <- nn_train_df[sample(seq_len(nrow(nn_train_df)), 1800), , drop = FALSE]
  }

  # Build formula safely without string parsing issues
  nn_formula <- stats::reformulate(colnames(train_xs), response = "log_price")

  # Shallow network to keep runtime reasonable
  nn_manual <- neuralnet::neuralnet(
    nn_formula,
    data = nn_train_df,
    hidden = c(4),
    linear.output = TRUE,
    stepmax = 2e5,
    threshold = 0.05
  )

  manual_pred <- neuralnet::compute(nn_manual, nn_test_df[, colnames(train_xs), drop = FALSE])$net.result
  manual_metrics <- compute_metrics(manual_pred, nn_test_df$log_price) |>
    dplyr::mutate(model = "neuralnet_manual")

  saveRDS(nn_manual, file.path(nn_dir, "nn_log_price_neuralnet.rds"))
  readr::write_delim(
    dplyr::bind_cols(test_data, tibble::tibble(pred_log_price = as.numeric(manual_pred))),
    file.path(nn_dir, "nn_log_price_neuralnet_predictions.csv"),
    delim = ";"
  )
} else {
  message("Package 'neuralnet' not available; skipping manual neuralnet model.")
}

# Compare NN models and record best by RMSE
metrics_list <- list(caret_metrics)
if (!is.null(manual_metrics)) {
  metrics_list <- c(metrics_list, list(manual_metrics))
}

all_metrics <- dplyr::bind_rows(metrics_list)
readr::write_csv(all_metrics, file.path(nn_dir, "nn_log_price_metrics.csv"))

best_row <- all_metrics |>
  dplyr::filter(.metric == "rmse") |>
  dplyr::arrange(.estimate) |>
  dplyr::slice(1)

best_label <- paste0(
  "Best NN model: ", best_row$model,
  " with RMSE = ", round(best_row$.estimate, 4),
  " (lower is better)"
)
writeLines(best_label, con = file.path(nn_dir, "nn_best_model.txt"))

message(best_label)
message("Saved caret nnet model to: ", file.path(nn_dir, "nn_log_price_caret.rds"))
message("Saved NN metrics to: ", file.path(nn_dir, "nn_log_price_metrics.csv"))
if (!is.null(manual_metrics)) {
  message("Saved neuralnet model to: ", file.path(nn_dir, "nn_log_price_neuralnet.rds"))
}
