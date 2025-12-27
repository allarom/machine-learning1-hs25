# helpers.R
# Shared utilities for data loading and consistent train/test splitting.

# Return a stored split (train/test indices) or create it if missing.
# Splits are saved at data/processed/train_test_split.rds to keep all models
# on the same hold-out set.
get_train_test_split <- function(n_rows, prop = 0.8, seed = 42L) {
  split_path <- here::here("data", "processed", "train_test_split.rds")

  if (file.exists(split_path)) {
    split <- readRDS(split_path)
    if (!all(c("train_idx", "test_idx") %in% names(split))) {
      stop("Stored split is invalid. Delete ", split_path, " to regenerate.")
    }
    if ((length(split$train_idx) + length(split$test_idx)) != n_rows) {
      stop(
        "Stored split length (", length(split$train_idx) + length(split$test_idx),
        ") does not match data rows (", n_rows, "). Delete ",
        split_path, " to regenerate."
      )
    }
    return(split)
  }

  set.seed(seed)
  n_train <- floor(prop * n_rows)
  train_idx <- sort(sample(seq_len(n_rows), size = n_train, replace = FALSE))
  test_idx <- setdiff(seq_len(n_rows), train_idx)

  split <- list(
    train_idx = train_idx,
    test_idx = test_idx,
    prop = prop,
    seed = seed
  )

  saveRDS(split, split_path)
  split
}

# Convenience helper to apply a split to a data frame
apply_split <- function(data, split) {
  list(
    train = data[split$train_idx, , drop = FALSE],
    test = data[split$test_idx, , drop = FALSE]
  )
}
