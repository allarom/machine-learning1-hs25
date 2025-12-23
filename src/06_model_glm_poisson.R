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
model_data <- cars |>
  dplyr::transmute(
    # response (count-like)
    milage_k_count = pmax(0L, as.integer(milage_k)),
    
    # predictors
    age,
    accident_bin,
    brand,
    fuel_type,
    transmission,
    ext_col,
    int_col
  ) |>
  tidyr::drop_na()

# -------------------------------------------------------------------
# 3. Fit Poisson GLM (log link)
# -------------------------------------------------------------------
glm_poisson <- stats::glm(
  milage_k_count ~ age + accident_bin + brand + fuel_type + transmission + ext_col + int_col,
  data = model_data,
  family = poisson(link = "log")
)

glm_sum <- summary(glm_poisson)

# Some compact diagnostics / performance-ish numbers
pseudo_r2 <- 1 - (glm_poisson$deviance / glm_poisson$null.deviance)

# dispersion check ( > 1 suggests overdispersion)
dispersion <- sum(stats::residuals(glm_poisson, type = "pearson")^2) / glm_poisson$df.residual

message("Poisson GLM fitted with ", length(coef(glm_poisson)), " coefficients.")
message("AIC: ", round(stats::AIC(glm_poisson), 2))
message("Pseudo-R² (deviance-based): ", round(pseudo_r2, 3))
message("Dispersion (Pearson): ", round(dispersion, 3))

# -------------------------------------------------------------------
# 4. Add predictions back to data (useful for plots / diagnostics)
# -------------------------------------------------------------------
model_data <- model_data |>
  dplyr::mutate(
    pred_milage_k = stats::predict(glm_poisson, type = "response"),
    resid_milage_k = milage_k_count - pred_milage_k
  )

# keep model and model_data in the environment for the report
assign("glm_poisson", glm_poisson, envir = .GlobalEnv)
assign("glm_poisson_data", model_data, envir = .GlobalEnv)
