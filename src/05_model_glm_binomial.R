# 05_model_glm_binomial.R


# ===========================================================
# Generalized Linear Model (GLM) - Binomial Family
# ===========================================================
# Purpose:
# Predict accident history of used cars using a binomial GLM.
# The model estimates the likelihood that a car has had at least
# one reported accident based on vehicle characteristics.

# load packages
source(here::here("src", "00_load_packages.R"))

# Define file paths and names
input_path <- here::here("data", "processed", "used_cars_features.csv")

# Read the dataset from CSV file
data <- readr::read_delim(input_path, delim = ";", show_col_types = FALSE)

# Fit the GLM
glm.car <- glm(
  accident_bin ~ brand + age + milage_k + fuel_type + transmission + price_dollar + horsepower + cylinders,
  data = data,
  family = "binomial"
)

summary(glm.car)

# Call:
#   glm(formula = accident_bin ~ brand + age + milage_k + fuel_type + 
#         transmission + price_dollar + horsepower + cylinders, family = "binomial", 
#       data = data)
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)             -7.565e-01  4.880e-01  -1.550   0.1211    
# brandAlfa               -1.102e+00  1.113e+00  -0.990   0.3221    
# brandAudi                2.338e-01  3.826e-01   0.611   0.5412    
# brandBentley             7.038e-02  7.604e-01   0.093   0.9263    
# brandBMW                -1.410e-01  3.655e-01  -0.386   0.6997    
# brandBuick               3.798e-01  6.019e-01   0.631   0.5280    
# brandCadillac            4.799e-02  4.195e-01   0.114   0.9089    
# brandChevrolet          -2.149e-01  3.804e-01  -0.565   0.5721    
# brandChrysler           -6.488e-02  5.726e-01  -0.113   0.9098    
# brandDodge               8.623e-02  4.292e-01   0.201   0.8408    
# brandFerrari             1.980e+00  1.275e+00   1.553   0.1205    
# brandFord               -8.352e-02  3.679e-01  -0.227   0.8204    
# brandGenesis            -1.373e+00  1.095e+00  -1.254   0.2097    
# brandGMC                 1.849e-01  4.318e-01   0.428   0.6685    
# brandHonda              -2.186e-01  4.732e-01  -0.462   0.6441    
# brandHummer              3.534e-01  6.157e-01   0.574   0.5660    
# brandHyundai             1.307e-01  4.424e-01   0.295   0.7677    
# brandINFINITI           -2.661e-01  4.589e-01  -0.580   0.5621    
# brandJaguar             -2.984e-02  5.020e-01  -0.059   0.9526    
# brandJeep               -3.128e-01  4.153e-01  -0.753   0.4513    
# brandKia                -1.944e-01  4.982e-01  -0.390   0.6964    
# brandLamborghini        -1.127e+01  3.235e+02  -0.035   0.9722    
# brandLand               -7.250e-01  4.439e-01  -1.633   0.1025    
# brandLexus               3.271e-01  3.909e-01   0.837   0.4027    
# brandLincoln            -1.301e-01  4.753e-01  -0.274   0.7843    
# brandMaserati           -1.064e+00  7.121e-01  -1.494   0.1352    
# brandMazda              -4.210e-02  5.076e-01  -0.083   0.9339    
# brandMercedes-Benz       8.343e-02  3.713e-01   0.225   0.8222    
# brandMINI               -1.140e+00  6.409e-01  -1.778   0.0754 .  
# brandMitsubishi          1.051e-01  5.975e-01   0.176   0.8604    
# brandNissan              1.616e-01  4.075e-01   0.396   0.6918    
# brandPontiac            -3.582e-01  6.867e-01  -0.522   0.6019    
# brandPorsche             2.548e-01  4.055e-01   0.629   0.5297    
# brandRAM                 3.967e-01  4.589e-01   0.864   0.3873    
# brandRivian             -1.300e+01  3.749e+02  -0.035   0.9723    
# brandSubaru              2.139e-01  4.458e-01   0.480   0.6313    
# brandTesla              -1.094e+00  8.824e-01  -1.240   0.2150    
# brandToyota             -1.868e-01  3.866e-01  -0.483   0.6289    
# brandVolkswagen         -3.929e-01  4.792e-01  -0.820   0.4123    
# brandVolvo              -5.475e-01  5.440e-01  -1.006   0.3142    
# age                     -1.164e-02  1.190e-02  -0.978   0.3281    
# milage_k                 6.995e-03  1.150e-03   6.083 1.18e-09 ***
# fuel_typeE85 Flex Fuel  -1.023e-01  3.112e-01  -0.329   0.7424    
# fuel_typeElectric       -1.286e+00  6.523e-01  -1.972   0.0486 *  
# fuel_typeGasoline       -1.889e-01  2.541e-01  -0.743   0.4574    
# fuel_typeHybrid         -2.298e-01  3.338e-01  -0.689   0.4911    
# fuel_typePlug-In Hybrid  4.227e-01  4.819e-01   0.877   0.3803    
# transmissionManual      -2.091e-01  1.606e-01  -1.302   0.1929    
# price_dollar            -2.169e-05  3.845e-06  -5.642 1.68e-08 ***
# horsepower               1.829e-03  8.796e-04   2.079   0.0376 *  
# cylinders               -3.470e-02  5.214e-02  -0.665   0.5058    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 3658.7  on 3064  degrees of freedom
# Residual deviance: 3306.0  on 3014  degrees of freedom
# AIC: 3408
# 
# Number of Fisher Scoring iterations: 14


# ===========================================================
# Exponentiate key coefficients for interpretation
# ===========================================================

# Mileage (milage_k)
#
exp(coef(glm.car)["milage_k"]*10)
# 1.072454
# Cars with 10,000 more miles have 7.2% higher odds of having an accident, holding all other variables constant.
# Interpretation: As expected, vehicles with higher mileage are slightly more likely to have been in an accident, reflecting increased usage and wear.

# Price (price_dollar)
#
exp(coef(glm.car)["price_dollar"] * 1000)
# 0.9785393
1-exp(coef(glm.car)["price_dollar"] * 1000)
# 0.02146066 
# For each additional $1,000 in price, the odds of a reported accident decrease by ~2.1%, holding all other variables constant.
# Interpretation: Higher-priced vehicles tend to have fewer accidents. This could reflect better build quality, more careful owners, or newer age.

# Horsepower (horsepower)
#
exp(coef(glm.car)["horsepower"] * 50)
# 1.095754
# Cars with 50 more horsepower have ~9.5% higher odds of an accident, holding other factors constant.
# Interpretation: More powerful vehicles are slightly more likely to have an accident history, possibly due to higher risk behavior or more spirited driving.

# Fuel Type – Electric
#
exp(coef(glm.car)["fuel_typeElectric"])
# 0.2762555
1-exp(coef(glm.car)["fuel_typeElectric"])
# 0.7237445
# Electric cars have 72% lower odds of a reported accident compared to the reference fuel type.
# Interpretation: This is a strong and statistically significant effect. However, looking at the dataset:

aggregate(data$age, list(data$fuel_type), FUN=mean)
# Group.1         x
# 1         Diesel  9.949495
# 2  E85 Flex Fuel 11.491935
# 3       Electric  4.612403
# 4       Gasoline 10.867717
# 5         Hybrid  5.978571
# 6 Plug-In Hybrid  5.393939

# Electric vehicles: 4.6 years (much newer than others)
# The effect is partly explained by electric cars being newer. Newer cars are less likely to have an accident.
# The coefficient captures both technology type and age effects.


# Brands
# 
# Most brand coefficients are not statistically significant.
# Some extreme cases (e.g., brandLamborghini or brandRivian) have huge standard errors due to very few observations.
# Interpretation: For most brands, accident risk does not differ significantly from the reference brand when controlling for other factors.

# Transmission
# 
# transmissionManual has a slightly negative coefficient, indicating a minor decrease in accident odds, but it is not statistically significant.
# Interpretation: Manual vs. automatic does not meaningfully affect accident risk in this dataset.

# Other variables
# 
# age and cylinders have small and non-significant effects.
# Interpretation: These factors do not strongly influence the probability of a reported accident in this model.

# Mileage vs Predicted Probability
# Predicted probabilities from the GLM
data$pred_prob <- predict(glm.car, type = "response")

# Plot predicted probability of accident vs. mileage (in 1000 miles)
ggplot(data, aes(x = milage_k, y = pred_prob)) +
  geom_point(alpha = 0.4, color = "blue") +       # scatter points
  geom_smooth(method = "loess", color = "red", se = TRUE) +  # smooth trend
  labs(
    title = "Predicted Probability of Accident vs Mileage",
    x = "Mileage (in 1000 miles)",
    y = "Predicted Probability of Accident"
  ) +
  theme_minimal()

# Plot predicted Probability of Accident vs Horsepower
ggplot(data, aes(x = horsepower, y = pred_prob)) +
  geom_point(alpha = 0.3, color = "darkgreen", size = 1.5) +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  labs(
    title = "Predicted Probability of Accident vs Horsepower",
    x = "Horsepower",
    y = "Predicted Probability of Accident"
  ) +
  theme_minimal()

# Tabular summary
# Variable        Odds Ratio	  % Change	p-value
# Mileage (10k)   1.072	        +7.2%	    <0.001
# Price ($1k)	    0.979	        -2.1%	    <0.001
# Horsepower (50)	1.096	        +9.6%	    0.038
# Fuel: Electric	0.276	        -72.4%	  0.049

# Conclusions
# 
# High mileage slightly increases the probability of a reported accident.
# Higher vehicle price reduces the probability of accidents modestly.
# Electric vehicles have substantially lower odds of accidents, but this effect is partly due to their newer age.
# Most brands, transmission types, and other features do not significantly alter accident likelihood.
# Horsepower shows a non-linear effect: accident risk increases up to around 250 hp and then decreases
# for higher horsepower vehicles. The current linear GLM does not fully capture this pattern, so the model may
# oversimplify the effect of horsepower. Overall, the GLM provides interpretable insights into which factors are associated
# with accident history.



