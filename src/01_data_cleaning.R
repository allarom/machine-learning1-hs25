# 01_data_cleaning.R
source("src/00_load_packages.R")


# 1) Load data -------------------------------------------------------------
raw_path <- "data/raw/used_cars.csv"
cars_raw <- readr::read_csv(raw_path, show_col_types = FALSE)

# Quick look
glimpse(cars_raw)
summary(cars_raw)


# ==============================================
# Data Cleaning and Preprocessing for Used Cars
# ==============================================
# This script reads the raw used cars dataset, cleans and transforms columns,
# handles outliers and rare categories, and prepares the dataset for predictive modeling.

# Define file paths and names
path='C:/Users/tashi/Hochschule Luzern/2. Semester/W.MPM02_Applied Machine Learning and Predictive Modelling 1/group_work/'
input_name='used_cars.csv'
output_name='used_cars_cleaned.csv'

# Read the dataset from CSV file
data <- read.csv(file.path(path, input_name),sep = ';')

# Keep only vehicles with a clean title to ensure reliable pricing data
data <- subset(data, clean_title == "Yes")

# Drop 'model' column due to high cardinality and sparsity, which can destabilize models
length(unique(data$model)) # 1679 unique models in 3413 rows
data$model <- NULL

# Extract numeric horsepower from 'engine' text and create a new 'horsepower' column
has_hp <- grepl("(\\d+\\.?\\d*)HP", data$engine, perl = TRUE)
data$horsepower <- NA_real_
data$horsepower[has_hp] <- as.numeric(gsub(".*?(\\d+\\.?\\d*)HP.*", "\\1", data$engine[has_hp], perl = TRUE))

# Extract number of cylinders from 'engine' text and create a new 'cylinders' column
has_cyl <- grepl("\\d+\\s*Cylinder", data$engine, perl = TRUE)
data$cylinders <- NA_real_
data$cylinders[has_cyl] <- as.numeric(gsub(".*?(\\d+)\\s*Cylinder.*", "\\1", data$engine[has_cyl], perl = TRUE))

# Identify electric vehicles and set cylinders to 0
is_electric <- grepl("Electric Motor Electric Fuel System", data$engine, ignore.case = TRUE)
data$cylinders[is_electric] <- 0

# Set fuel type to 'Electric' for identified electric vehicles
data$fuel_type[is_electric] <- 'Electric'

# Convert 'milage' from text (e.g., "40,000 mi.") to numeric miles
data$milage <- as.numeric(gsub(",", "", gsub(" mi\\.", "", data$milage)))

# Clean 'price' column by removing '$' and commas, convert to numeric, then rename to 'price_dollar'
data$price <- as.numeric(gsub(",", "", gsub("\\$", "", data$price)))
colnames(data)[colnames(data) == "price"] <- "price_dollar"

# Remove rows with missing horsepower or cylinders to avoid issues in modeling
data <- subset(data, !is.na(horsepower) & !is.na(cylinders))

# Keep only brands with at least 8 rows to ensure sufficient data per brand, retaining rare but important brands like Ferrari
brand_counts <- table(data$brand)
valid_brands <- names(brand_counts[brand_counts >= 8])
data <- data[data$brand %in% valid_brands, ]

# Simplify transmissions into Manual or Automatic; CVT and overdrive switches are treated as Automatic
data$transmission <- ifelse(grepl("M/T", data$transmission), "Manual", "Automatic")
data$transmission <- factor(data$transmission)

# Replace corrupted color value 'â\200“' with "Unknown"
data$int_col[data$int_col == "â\200“"] <- "Unknown"

# Drop rows with extreme price outliers (2954083 and 1599000) to avoid skewing the model
data <- data[!data$price_dollar %in% c(2954083, 1599000), ]

# Write the cleaned dataset to a CSV file
write.table(data, file.path(path, output_name), sep = ";", row.names = FALSE, quote = TRUE)
