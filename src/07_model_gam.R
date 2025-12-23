# 07_model_gam.R

# ===========================================================
# Generalised Additive Model (GAM) - Binomial Family
# ===========================================================
# Purpose:
# Predict the probability that a car has had at least one reported accident
# using a GAM. Smooth terms are used to capture potential non-linear
# relationships (e.g., mileage, price, horsepower, and age).

# load packages
source(here::here("src", "00_load_packages.R"))

# Define file paths and names
input_path <- here::here("data", "processed", "used_cars_features.csv")

# Read the dataset from CSV file
data <- readr::read_delim(input_path, delim = ";", show_col_types = FALSE)


# ===========================================================
# Fit GAM Model
# ===========================================================

gam.car <- gam(
  accident_bin ~ 
    brand +                     # categorical brand effects (parametric)
    fuel_type +                 # fuel type differences in accident risk
    transmission +              # automatic vs manual
    cylinders +                 # linear effect (no evidence of non-linearity)
    s(milage_k) +               # smooth: mileage (in 1,000 miles)
    s(price_dollar) +           # smooth: vehicle price
    s(horsepower) +             # smooth: engine power
    s(age),                     # smooth: vehicle age
  data = data,
  family = "binomial",          # binary outcome: accident vs no accident
  method = "REML"               # penalized likelihood for smoothness selection
)

# ===========================================================
# Summary of Model
# ===========================================================

summary(gam.car)

# ===========================================================
# Interpretation of Smooth Terms
# ===========================================================
# 1. Significant smooth effects at 5% threshold:
#    - s(milage_k)   : highly significant, p < 2e-16
#    - s(price_dollar): significant, p = 0.00284
# 2. Non-significant smooth terms at 5%:
#    - s(horsepower) : p = 0.216
#    - s(age)        : p = 0.078
# 3. Linear vs Non-linear:
#    - None of the smooth terms have edf ≈ 1, so all are treated as non-linear.
#    - s(price_dollar) has the lowest edf (2.266), so its effect is closest to linear.
# 4. Most complex smooth term:
#    - s(milage_k) with edf = 5.696, indicating a strongly non-linear relationship.
#      Mileage has the most complex influence on accident probability.

# ===========================================================
# Interpretation of Parametric Terms
# ===========================================================
# - Categorical terms (brand, fuel_type, transmission) are estimated as in logistic regression.
# - Example: brandMINI has a coefficient of -1.305 (p = 0.044), suggesting lower log-odds
#   of accident compared to the reference brand, controlling for other factors.
# - Cylinders are treated linearly; coefficient not significant, suggesting minimal effect.

# ===========================================================
# Overall Model Insights
# ===========================================================
# - Mileage and price are the main predictors with significant non-linear effects.
# - Horsepower and age do not show statistically significant smooth effects, although they are modeled non-linearly.
# - The model captures both categorical effects (brand, fuel type, transmission) and complex non-linear relationships.
# - The most intricate effect is observed for mileage, indicating that accident probability changes non-linearly with kilometers driven.
# - This GAM approach allows a flexible yet interpretable way to model accident risk, making it suitable for predictive and explanatory purposes.

