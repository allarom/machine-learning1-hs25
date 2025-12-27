# 08_model_svm.R
# Support Vector Machine regression for log_price using e1071 and kernlab.
# Loads engineered features, tunes small radial-kernel grids, evaluates on a hold-out
# test set, and saves both fitted models plus metrics/predictions for comparison.

source("src/00_load_packages.R")
source("src/helpers.R")

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

split <- get_train_test_split(nrow(model_data), prop = 0.8, seed = 42)
parts <- apply_split(model_data, split)
train_data <- parts$train
test_data <- parts$test

compute_metrics <- function(pred, truth) {
  results_tbl <- tibble::tibble(truth = truth, estimate = as.numeric(pred))
  metrics <- yardstick::metric_set(yardstick::rmse, yardstick::mae, yardstick::rsq)
  metrics(results_tbl, truth = truth, estimate = estimate)
}

# Shared output directory for all SVM artifacts
svm_dir <- here::here("report", "models", "svm")
dir.create(svm_dir, recursive = TRUE, showWarnings = FALSE)

# Radial kernel SVM (e1071) with a compact cost/gamma grid (kept small for speed);
# 3-fold cross validation inside tune() selects the best combo on the training split.
tune_result <- e1071::tune(
  e1071::svm,
  log_price ~ .,
  data = train_data,
  kernel = "radial",
  ranges = list(
    cost = c(0.5, 1, 2, 4, 8),
    gamma = c(0.01, 0.05, 0.1, 0.2)
  ),
  tunecontrol = e1071::tune.control(cross = 3)
)

best_model <- tune_result$best.model
message(
  "Best SVM parameters: cost=", best_model$cost,
  ", gamma=", best_model$gamma,
  " (CV RMSE=", round(sqrt(tune_result$best.performance), 4), ")"
)

# Visualize tuning surface (CV RMSE) and save to disk
tune_df <- tibble::as_tibble(tune_result$performances) |>
  dplyr::rename(cv_mse = error) |>
  dplyr::mutate(cv_rmse = sqrt(cv_mse))

tune_plot <- ggplot2::ggplot(
  tune_df,
  ggplot2::aes(x = factor(cost), y = factor(gamma), fill = cv_rmse)
) +
  ggplot2::geom_tile(color = "white") +
  ggplot2::scale_fill_gradient(low = "#22C55E", high = "#EF4444", name = "CV RMSE") +
  ggplot2::labs(
    title = "SVM radial tuning (e1071)",
    x = "cost",
    y = "gamma"
  ) +
  ggplot2::theme_minimal(base_size = 12)

ggplot2::ggsave(
  filename = file.path(svm_dir, "svm_tuning_heatmap.png"),
  plot = tune_plot,
  width = 6,
  height = 5,
  dpi = 300
)

e1071_pred <- stats::predict(best_model, newdata = test_data)
e1071_metrics <- compute_metrics(e1071_pred, test_data$log_price) |>
  dplyr::mutate(model = "e1071_radial")

saveRDS(best_model, file.path(svm_dir, "svm_log_price_e1071.rds"))
readr::write_delim(
  dplyr::bind_cols(test_data, tibble::tibble(pred_log_price = as.numeric(e1071_pred))),
  file.path(svm_dir, "svm_log_price_e1071_predictions.csv"),
  delim = ";"
)

# kernlab SVM via caret::train (radial basis) on a compact C/sigma grid;
# 3-fold cross validation inside caret selects the best combo on the training split.
ctrl <- caret::trainControl(method = "cv", number = 3)
tune_grid <- expand.grid(
  C = c(1, 2, 4, 8),
  sigma = c(0.01, 0.05, 0.1, 0.2)
)

kern_model <- caret::train(
  log_price ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = tune_grid
)

message(
  "Best kernlab parameters: C=", kern_model$bestTune$C,
  ", sigma=", kern_model$bestTune$sigma,
  " (CV RMSE=", round(min(kern_model$results$RMSE), 4), ")"
)

# Visualize caret tuning surface (CV RMSE) and save
caret_tune_df <- tibble::as_tibble(kern_model$results) |>
  dplyr::rename(cv_rmse = RMSE)

caret_tune_plot <- ggplot2::ggplot(
  caret_tune_df,
  ggplot2::aes(x = factor(C), y = factor(sigma), fill = cv_rmse)
) +
  ggplot2::geom_tile(color = "white") +
  ggplot2::scale_fill_gradient(low = "#22C55E", high = "#EF4444", name = "CV RMSE") +
  ggplot2::labs(
    title = "SVM radial tuning (caret/kernlab)",
    x = "C",
    y = "sigma"
  ) +
  ggplot2::theme_minimal(base_size = 12)

ggplot2::ggsave(
  filename = file.path(svm_dir, "svm_tuning_heatmap_caret.png"),
  plot = caret_tune_plot,
  width = 6,
  height = 5,
  dpi = 300
)

kern_pred <- stats::predict(kern_model, newdata = test_data)
kern_metrics <- compute_metrics(kern_pred, test_data$log_price) |>
  dplyr::mutate(model = "kernlab_radial")

saveRDS(kern_model, file.path(svm_dir, "svm_log_price_kernlab.rds"))
readr::write_delim(
  dplyr::bind_cols(test_data, tibble::tibble(pred_log_price = as.numeric(kern_pred))),
  file.path(svm_dir, "svm_log_price_kernlab_predictions.csv"),
  delim = ";"
)

# Compare models and record best by RMSE
all_metrics <- dplyr::bind_rows(e1071_metrics, kern_metrics)
readr::write_csv(all_metrics, file.path(svm_dir, "svm_log_price_metrics.csv"))

best_row <- all_metrics |>
  dplyr::filter(.metric == "rmse") |>
  dplyr::arrange(.estimate) |>
  dplyr::slice(1)

best_label <- paste0(
  "Best model: ", best_row$model,
  " with RMSE = ", round(best_row$.estimate, 4),
  " (lower is better)"
)
writeLines(best_label, con = file.path(svm_dir, "svm_best_model.txt"))

message(best_label)
message("Saved tuned e1071 model to: ", file.path(svm_dir, "svm_log_price_e1071.rds"))
message("Saved tuned kernlab model to: ", file.path(svm_dir, "svm_log_price_kernlab.rds"))
message("Saved metrics to: ", file.path(svm_dir, "svm_log_price_metrics.csv"))
