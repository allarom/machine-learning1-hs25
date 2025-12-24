# 03_exploratory_analysis.R
# Exploratory analysis of the engineered used car dataset.
# Generates core plots and saves them under report/plots.

source("src/00_load_packages.R")

data_path <- here::here("data", "processed", "used_cars_features.csv")
plot_dir <- here::here("report", "plots")

if (!file.exists(data_path)) {
  stop("Data file not found at: ", data_path)
}

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

cars <- readr::read_delim(data_path, delim = ";", show_col_types = FALSE) |>
  dplyr::mutate(
    brand = factor(brand),
    fuel_type = factor(fuel_type),
    transmission = factor(transmission),
    ext_col = factor(ext_col),
    int_col = factor(int_col),
    accident = factor(accident)
  )

message("Loaded ", nrow(cars), " rows and ", ncol(cars), " columns from used_cars_features.csv")

save_plot <- function(plot, filename, width = 9, height = 6) {
  out_path <- file.path(plot_dir, filename)
  ggplot2::ggsave(out_path, plot = plot, width = width, height = height, dpi = 320)
  message("Saved plot: ", out_path)
}

# Price distribution on the raw scale to show the heavy tail
price_hist_raw <- cars |>
  ggplot2::ggplot(ggplot2::aes(price_dollar)) +
  ggplot2::geom_histogram(bins = 45, fill = "#0EA5E9", color = "white", alpha = 0.9) +
  ggplot2::labs(
    title = "Used car prices (raw scale)",
    x = "Price (USD)",
    y = "Count"
  ) +
  ggplot2::coord_cartesian(xlim = c(0, 250000)) +
  ggplot2::scale_x_continuous(labels = scales::dollar_format()) +
  ggplot2::theme_minimal(base_size = 12)

# Price distribution on a log scale to tame heavy right tail
price_hist <- cars |>
  ggplot2::ggplot(ggplot2::aes(price_dollar)) +
  ggplot2::geom_histogram(bins = 45, fill = "#2563EB", color = "white", alpha = 0.9) +
  ggplot2::coord_cartesian(xlim = c(NA, 250000)) +
  ggplot2::scale_x_log10(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = "Used car prices (log scale)",
    x = "Price (USD, log scale)",
    y = "Count"
  ) +
  ggplot2::theme_minimal(base_size = 12)

# Age vs price to highlight depreciation across fuels
age_price <- cars |>
  ggplot2::ggplot(ggplot2::aes(age, price_dollar, color = fuel_type)) +
  ggplot2::geom_jitter(alpha = 0.25, size = 1.2) +
  ggplot2::geom_smooth(se = FALSE, method = "loess") +
  ggplot2::scale_y_log10(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = "Depreciation by fuel type",
    x = "Age (years)",
    y = "Price (USD, log scale)",
    color = "Fuel type"
  ) +
  ggplot2::theme_minimal(base_size = 12)

# Price variation for the 12 most common brands
top_brands <- cars |>
  dplyr::count(brand, sort = TRUE) |>
  dplyr::slice_head(n = 12) |>
  dplyr::pull(brand)

brand_price <- cars |>
  dplyr::filter(brand %in% top_brands) |>
  dplyr::mutate(brand = forcats::fct_reorder(brand, price_dollar, .fun = median)) |>
  ggplot2::ggplot(ggplot2::aes(brand, price_dollar, fill = brand)) +
  ggplot2::geom_boxplot(outlier.alpha = 0.15, show.legend = FALSE) +
  ggplot2::scale_y_log10(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = "Price spread for top-selling brands",
    x = "Brand (ordered by median price)",
    y = "Price (USD, log scale)"
  ) +
  ggplot2::coord_flip() +
  ggplot2::theme_minimal(base_size = 12)

# Mileage vs price to visualize wear-and-tear discounts
milage_price <- cars |>
  ggplot2::ggplot(ggplot2::aes(milage_k, price_dollar, color = transmission)) +
  ggplot2::geom_point(alpha = 0.25, size = 1.2) +
  ggplot2::geom_smooth(se = FALSE, method = "loess") +
  ggplot2::scale_y_log10(labels = scales::dollar_format()) +
  ggplot2::coord_cartesian(xlim = c(0, 250)) + # trim extreme mileage outliers for clearer smoothing
  ggplot2::labs(
    title = "Mileage impact by transmission",
    x = "Mileage (thousands of miles)",
    y = "Price (USD, log scale)",
    color = "Transmission"
  ) +
  ggplot2::theme_minimal(base_size = 12)

# Powertrain view: horsepower vs price, colored by transmission
power_price <- cars |>
  ggplot2::ggplot(ggplot2::aes(horsepower, price_dollar, color = transmission)) +
  ggplot2::geom_point(alpha = 0.25, size = 1.2) +
  ggplot2::geom_smooth(se = FALSE, method = "loess") +
  ggplot2::scale_y_log10(labels = scales::dollar_format()) +
  ggplot2::coord_cartesian(xlim = c(0, 700)) + # limit extreme HP outliers to keep smoother stable
  ggplot2::labs(
    title = "Horsepower premium",
    x = "Horsepower",
    y = "Price (USD, log scale)",
    color = "Transmission"
  ) +
  ggplot2::theme_minimal(base_size = 12)

# Accident history effect
accident_price <- cars |>
  ggplot2::ggplot(ggplot2::aes(accident, price_dollar, fill = accident)) +
  ggplot2::geom_boxplot(outlier.alpha = 0.2, show.legend = FALSE) +
  ggplot2::scale_y_log10(labels = scales::dollar_format()) +
  ggplot2::labs(
    title = "Accident history and price",
    x = "Accident history",
    y = "Price (USD, log scale)"
  ) +
  ggplot2::theme_minimal(base_size = 12)

save_plot(price_hist_raw, "price_distribution_raw.png")
save_plot(price_hist, "price_distribution.png")
save_plot(age_price, "age_vs_price.png")
save_plot(brand_price, "brand_price_boxplots.png")
save_plot(milage_price, "milage_vs_price.png")
save_plot(power_price, "horsepower_vs_price.png")
save_plot(accident_price, "accident_vs_price.png")

message("Exploratory plots written to: ", plot_dir)
