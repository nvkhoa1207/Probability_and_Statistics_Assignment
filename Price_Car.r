#======================== LOAD ALL LIBRARIES ========================
library(questionr)
library(corrplot)
library(car)
library(dunn.test)
library(lmtest)
library(tidyr)
library(dplyr)
library(ggplot2)
library(randomForest)
#========================1. Data Preprocessing=================
# Load the dataset
car_data <- read.csv("~/Desktop/studying/xstk/Elite Sports Cars in Data.csv")
head(car_data, 10)

# Remove unnecessary columns: Log_Price, Log_Mileage, and Modification
car_data <- car_data[, !names(car_data) %in% c("Log_Price", "Log_Mileage", "Modification")]
# Check missing values in the dataset
freq.na(car_data)

# Create a function to detect outliers
check_outliers <- function(data) {
  
  # Select only numeric columns
  num <- data[, sapply(data, is.numeric), drop = FALSE]
  
  # Identify outliers using the IQR method
  out <- sapply(num, function(x) {
    # Calculate the first quartile (Q1)
    Q1 <- quantile(x, 0.25)
    # Calculate the third quartile (Q3)
    Q3 <- quantile(x, 0.75)
    # Calculate the Interquartile Range (IQR)
    IQR <- Q3 - Q1
    
    # Count the number of outliers (values outside the 1.5 * IQR range)
    sum(x < Q1 - 1.5 * IQR | x > Q3 + 1.5 * IQR)
  })
  
  # Calculate the total number of values for each column
  total <- sapply(num, function(x) sum(!is.na(x)))
  
  # Calculate the percentage of outliers
  percent <- round(out / total * 100, 2)
  
  # Return a data frame with the count and percentage of outliers
  data.frame(outliers = out, percent = percent)
}

# Check for outliers in the car_data dataset
check_outliers(car_data)

# Display the frequency table of the 'Production_Units' column
table(car_data$Production_Units)

# Analysis of outliers:
# -> If there are few outliers, keep them unchanged.
# -> If there are many outliers, consider replacing them with the mean or median.
# -> Since the variable is important, outliers should not be removed.
# -> RECOMMENDATION: DO NOT HANDLE OUTLIERS.

# Convert categorical variables to factors
car_data$Brand <- as.factor(car_data$Brand)
car_data$Model <- as.factor(car_data$Model)
car_data$Country <- as.factor(car_data$Country)
car_data$Condition <- as.factor(car_data$Condition)
car_data$Fuel_Type <- as.factor(car_data$Fuel_Type)
car_data$Drivetrain <- as.factor(car_data$Drivetrain)
car_data$Transmission <- as.factor(car_data$Transmission)
car_data$Popularity <- as.factor(car_data$Popularity)
car_data$Safety_Rating <- as.factor(car_data$Safety_Rating)
car_data$Market_Demand <- as.factor(car_data$Market_Demand)


#===============================2. Descriptive Statistics========================

# Select continuous variables for descriptive statistics
cons_var <- car_data[, c("Year", "Engine_Size", "Horsepower", "Torque",
                         "Weight", "Top_Speed", "Acceleration_0_100", "Fuel_Efficiency",
                         "Price", "Mileage", "Insurance_Cost", "Production_Units", 
                         "Number_of_Owners")]

# Define a function to compute descriptive statistics for a numeric vector
describe_func <- function(x) {
  c(
    xtb = mean(x),  # Mean
    std = sd(x),    # Standard deviation
    med = median(x),# Median
    Q1 = quantile(x, probs = 0.25), # 1st Quartile
    Q3 = quantile(x, probs = 0.75), # 3rd Quartile
    GTNN = min(x),  # Minimum value
    GTLN = max(x)   # Maximum value
  )
}

# Apply the descriptive statistics function to each column of continuous variables
apply(cons_var, 2, describe_func)

# Select categorical variables for descriptive statistics
dis_var <- car_data[c("Brand", "Model", "Country", "Condition",
                      "Fuel_Type", "Drivetrain", "Transmission",
                      "Popularity", "Safety_Rating", "Market_Demand")]

# Display summary statistics for categorical variables
summary(dis_var)

#===============================3. Data Visualization========================

# Histogram of Car Prices
hist(car_data$Price,
     xlab = "Price",
     main = "Histogram of Car Prices",
     col = "lightgreen",
     labels = TRUE)

# Boxplots of Car Prices by Different Categories
boxplot(Price ~ Brand, data = car_data,
        main = "Boxplot of Car Prices for Brand",
        col = 2:5)

boxplot(Price ~ Model, data = car_data,
        main = "Boxplot of Car Prices for Model",
        col = 2:5)

boxplot(Price ~ Country, data = car_data,
        main = "Boxplot of Car Prices for Country",
        col = 2:4)

boxplot(Price ~ Condition, data = car_data,
        main = "Boxplot of Car Prices for Condition",
        col = 2:5)

boxplot(Price ~ Fuel_Type, data = car_data,
        main = "Boxplot of Car Prices for Fuel Type",
        col = 2:4)

boxplot(Price ~ Drivetrain, data = car_data,
        main = "Boxplot of Car Prices for Drivetrain",
        col = 2:4)

boxplot(Price ~ Transmission, data = car_data,
        main = "Boxplot of Car Prices for Transmission",
        col = 2:5)

boxplot(Price ~ Popularity, data = car_data,
        main = "Boxplot of Car Prices for Popularity",
        col = 2:4)

boxplot(Price ~ Safety_Rating, data = car_data,
        main = "Boxplot of Car Prices for Safety Rating",
        col = 2:5)

boxplot(Price ~ Market_Demand, data = car_data,
        main = "Boxplot of Car Prices for Market Demand",
        col = 2:4)

# Scatter Plots for Price vs Various Numeric Variables
par(mfrow = c(1, 3))  # Arrange plots in a 1x3 grid

plot(car_data$Year, car_data$Price,
     xlab = "Year", ylab = "Price", main = "Price & Year",
     pch = 20, cex = 0.5, col = "red")

plot(car_data$Engine_Size, car_data$Price,
     xlab = "Engine Size", ylab = "Price", main = "Price & Engine Size",
     pch = 20, cex = 0.5, col = "blue")

plot(car_data$Horsepower, car_data$Price,
     xlab = "Horsepower", ylab = "Price", main = "Price & Horsepower",
     pch = 20, cex = 0.5, col = "orange")

plot(car_data$Torque, car_data$Price,
     xlab = "Torque", ylab = "Price", main = "Price & Torque",
     pch = 20, cex = 0.5, col = "red")

plot(car_data$Weight, car_data$Price,
     xlab = "Weight", ylab = "Price", main = "Price & Weight",
     pch = 20, cex = 0.5, col = "blue")

plot(car_data$Top_Speed, car_data$Price,
     xlab = "Top Speed", ylab = "Price", main = "Price & Top Speed",
     pch = 20, cex = 0.5, col = "orange")

plot(car_data$Acceleration_0_100, car_data$Price,
     xlab = "Acceleration", ylab = "Price", main = "Price & Acceleration",
     pch = 20, cex = 0.5, col = "red")

plot(car_data$Fuel_Efficiency, car_data$Price,
     xlab = "Fuel Efficiency", ylab = "Price", main = "Price & Fuel Efficiency",
     pch = 20, cex = 0.5, col = "blue")

plot(car_data$Mileage, car_data$Price,
     xlab = "Mileage", ylab = "Price", main = "Price & Mileage",
     pch = 20, cex = 0.5, col = "orange")

plot(car_data$Insurance_Cost, car_data$Price,
     xlab = "Insurance Cost", ylab = "Price", main = "Price & Insurance Cost",
     pch = 20, cex = 0.5, col = "red")

plot(car_data$Production_Units, car_data$Price,
     xlab = "Production Units", ylab = "Price", main = "Price & Production Units",
     pch = 20, cex = 0.5, col = "blue")

plot(car_data$Number_of_Owners, car_data$Price,
     xlab = "Number of Owners", ylab = "Price", main = "Price & Number of Owners",
     pch = 20, cex = 0.5, col = "blue")

par(mfrow = c(1, 1))  # Reset to a single plot

# Correlation Matrix for Continuous Variables
cor_data <- cor(cons_var)
corrplot(cor_data,
         method = "color",
         addCoef.col = TRUE,
         number.cex = 0.5)

# Barplots for Frequency of Categorical Variables
barplot(table(car_data$Brand),
        main = "Barplot of Car Brands",
        xlab = "Brand", ylab = "Frequency",
        col = "lightblue", las = 2)

barplot(table(car_data$Model),
        main = "Barplot of Car Models",
        xlab = "Model", ylab = "Frequency",
        col = "lightgreen", las = 2)

barplot(table(car_data$Country),
        main = "Barplot of Car Countries",
        xlab = "Country", ylab = "Frequency",
        col = "lightcoral", las = 2)

barplot(table(car_data$Condition),
        main = "Barplot of Car Conditions",
        xlab = "Condition", ylab = "Frequency",
        col = "lightpink", las = 2)

barplot(table(car_data$Fuel_Type),
        main = "Barplot of Fuel Types",
        xlab = "Fuel Type", ylab = "Frequency",
        col = "lightyellow", las = 2)

barplot(table(car_data$Drivetrain),
        main = "Barplot of Car Drivetrains",
        xlab = "Drivetrain", ylab = "Frequency",
        col = "lightblue", las = 2)

barplot(table(car_data$Transmission),
        main = "Barplot of Car Transmissions",
        xlab = "Transmission", ylab = "Frequency",
        col = "lightgreen", las = 2)

barplot(table(car_data$Popularity),
        main = "Barplot of Car Popularity",
        xlab = "Popularity", ylab = "Frequency",
        col = "lightcoral", las = 2)

barplot(table(car_data$Safety_Rating),
        main = "Barplot of Car Safety Ratings",
        xlab = "Safety Rating", ylab = "Frequency",
        col = "lightpink", las = 2)

barplot(table(car_data$Market_Demand),
        main = "Barplot of Car Market Demand",
        xlab = "Market Demand", ylab = "Frequency",
        col = "lightyellow", las = 2)


#=========================3. Inferential Statistics===================
# METHOD 1: Two-Sample Test: Does the average price of Asian cars exceed that of American cars?

# Subset data for Asia and USA
Asia_data <- subset(car_data, Country == "Asia")
USA_data <- subset(car_data, Country == "USA")

# Check normality using QQ plot and Shapiro-Wilk test for Asia cars
qqnorm(Asia_data$Price)
qqline(Asia_data$Price)
shapiro.test(Asia_data$Price)

# Check normality for USA cars
qqnorm(USA_data$Price)
qqline(USA_data$Price)
shapiro.test(USA_data$Price)

# Test for equality of variances between the two groups
var.test(Asia_data$Price, USA_data$Price)

# Perform a two-sample t-test (assuming equal variance)
t.test(Asia_data$Price, USA_data$Price, var.equal = TRUE, alternative = "less")

# RECOMMENDATION: Since the data doesn't follow a normal distribution, use the Wilcoxon test instead of the t-test.
wilcox.test(Asia_data$Price, USA_data$Price, var.equal = TRUE, alternative = "less")

# METHOD 2: Two-Way ANOVA with Repeated Measures

# Hypothesis testing:
# 1. Does the factor Fuel_Type affect the car prices?
# 2. Does the factor Condition affect the car prices?
# 3. Does the interaction between Fuel_Type and Condition affect car prices?

# Test for equal variances using Levene's Test
leveneTest(Price ~ Fuel_Type * Condition, data = car_data)

# Create a list of subsets for the combinations of Fuel_Type and Condition
data_list <- list(
  data_1 = subset(car_data, Fuel_Type == "Diesel" & Condition == "new"),
  data_2 = subset(car_data, Fuel_Type == "Diesel" & Condition == "restored"),
  data_3 = subset(car_data, Fuel_Type == "Diesel" & Condition == "salvage"),
  data_4 = subset(car_data, Fuel_Type == "Diesel" & Condition == "used"),
  data_5 = subset(car_data, Fuel_Type == "Electric" & Condition == "new"),
  data_6 = subset(car_data, Fuel_Type == "Electric" & Condition == "restored"),
  data_7 = subset(car_data, Fuel_Type == "Electric" & Condition == "used"),
  data_8 = subset(car_data, Fuel_Type == "Petrol" & Condition == "new"),
  data_9 = subset(car_data, Fuel_Type == "Petrol" & Condition == "restored"),
  data_10 = subset(car_data, Fuel_Type == "Petrol" & Condition == "salvage"),
  data_11 = subset(car_data, Fuel_Type == "Petrol" & Condition == "used")
)

# Check normality for each subset using QQ plots and Shapiro-Wilk test
par(mfrow = c(3, 4))  # Set up the layout for multiple plots

for (i in 1:length(data_list)) {
  data <- data_list[[i]]
  
  # QQ plot for Price distribution
  qqnorm(data$Price, main = paste("QQ plot -", names(data_list)[i]))
  qqline(data$Price, col = "red")
  
  # Perform Shapiro-Wilk test for normality
  sw_test <- shapiro.test(data$Price)
  print(paste(names(data_list)[i], "Shapiro-Wilk p-value:", sw_test$p.value))
  
  # Output Shapiro-Wilk test results
  if (sw_test$p.value < 0.05) {
    print(paste(names(data_list)[i], "data does not follow a normal distribution."))
  } else {
    print(paste(names(data_list)[i], "data may follow a normal distribution."))
  }
}

# Variance is equal; however, normality is violated in some subsets (except data 6).
# However, due to the large sample size, ANOVA is still robust enough for use.
two_way_anova <- aov(Price ~ Fuel_Type * Condition, data = car_data)
summary(two_way_anova)

# Based on the ANOVA result, we conclude:
# - There is no significant difference in car prices based on Fuel_Type.
# - There is no significant difference in car prices based on Condition.
# - There is no significant interaction between Fuel_Type and Condition.

# Post-hoc analysis using Tukey HSD test for pairwise comparisons
tukey_result <- TukeyHSD(two_way_anova)
summary(tukey_result)



#METHOD 3: Kruskal-Wallis Test (for non-normally distributed data)
# Kruskal-Wallis test between Condition groups
kruskal_test_condition <- kruskal.test(Price ~ Condition, data = car_data)
print(kruskal_test_condition)

# Kruskal-Wallis test between Fuel_Type groups
kruskal_test_fuel <- kruskal.test(Price ~ Fuel_Type, data = car_data)
print(kruskal_test_fuel)

# Create a combined variable for Fuel_Type and Condition interaction
car_data$Fuel_Condition <- interaction(car_data$Fuel_Type, car_data$Condition)

# Perform Kruskal-Wallis test between the combined groups
kruskal_test_interaction <- kruskal.test(Price ~ Fuel_Condition, data = car_data)
print(kruskal_test_interaction)

# Interpretation:
# - The result shows no significant difference in mean prices between different fuel types.
# - There is no significant difference in mean prices between different conditions.
# - There is no significant interaction between Condition and Fuel_Type in terms of car price.

# Dunn's Test (post-hoc analysis) after Kruskal-Wallis test

# Perform Dunn's test for pairwise comparisons with Bonferroni correction
dunn_test_result <- dunn.test(car_data$Price, car_data$Fuel_Condition, method = "bonferroni")

# Print Dunn's test results
print(dunn_test_result)


#=====================LINEAR REGRESSION MODEL==========================

# Set seed for reproducibility
set.seed(123)

# Split the data into training (80%) and testing (20%) sets
train_index <- sample(1:nrow(car_data), 0.8 * nrow(car_data))

train_data <- car_data[train_index, ]
test_data <- car_data[-train_index, ]

# Fit a linear regression model with multiple predictors
model <- lm(Price ~ Brand + Model + Year + Country + Condition +
              Engine_Size + Horsepower + Torque + Weight + Top_Speed +
              Acceleration_0_100 + Fuel_Type + Drivetrain + Transmission +
              Fuel_Efficiency + CO2_Emissions + Mileage + Popularity +
              Safety_Rating + Number_of_Owners + Market_Demand + Insurance_Cost +
              Production_Units, data = train_data)

# Perform stepwise selection to find the best model
best_model <- step(model)

# Display the summary of the best model
summary(best_model)

# Plot diagnostic plots for the best model
par(mfrow = c(2, 2))
plot(best_model)

# Perform Durbin-Watson test to check for autocorrelation in residuals
dwtest(model)

# Check for multicollinearity using Variance Inflation Factor (VIF)
vif(best_model)

# Predict the car prices on the test dataset
predicted_price <- predict(model, newdata = test_data)

# Create a comparison dataframe for actual vs predicted prices
comparison <- data.frame(
  Actual = test_data$Price,
  Predicted = predicted_price
)

# Display the first 10 rows of the comparison
head(comparison, 10)

# Compute model evaluation metrics: MAE, RMSE, and R²
mae <- mean(abs(comparison$Actual - comparison$Predicted))  # Mean Absolute Error
rmse <- sqrt(mean((comparison$Actual - comparison$Predicted)^2))  # Root Mean Squared Error
ss_res <- sum((comparison$Actual - comparison$Predicted)^2)  # Residual sum of squares
ss_tot <- sum((comparison$Actual - mean(comparison$Actual))^2)  # Total sum of squares
r_squared <- 1 - ss_res / ss_tot  # R-squared

# Print the evaluation metrics
print(mae)
print(rmse)
print(r_squared)

# Predict the car prices on the test set with confidence intervals
predicted_price <- predict(best_model, newdata = test_data, interval = "confidence")

# Create a dataframe with actual prices, predicted prices, and the confidence intervals
comparison <- data.frame(
  Actual = test_data$Price,
  Predicted = predicted_price[, 1],  # Predicted value
  Lower_CI = predicted_price[, 2],   # Lower bound of confidence interval
  Upper_CI = predicted_price[, 3]    # Upper bound of confidence interval
)

# Display the first 10 rows of the comparison with confidence intervals
head(comparison, 10)

# Create a scatter plot comparing actual and predicted prices with confidence intervals
ggplot(comparison, aes(x = Actual, y = Predicted)) + 
  geom_point(color = "blue", alpha = 0.6) + 
  geom_errorbar(aes(ymin = Lower_CI, ymax = Upper_CI), width = 0.2, color = "red") +
  theme_minimal() +
  labs(title = "Price Prediction with Confidence Interval", 
       x = "Actual Price", 
       y = "Predicted Price")
#========================= Density Plot: Actual vs Predicted (Linear Regression) =========================

# Convert to long format for density comparison
density_long <- comparison %>%
  select(Actual, Predicted) %>% 
  pivot_longer(cols = everything(),
               names_to = "Type",          # "Actual" or "Predicted"
               values_to = "Value")        # Combined values for density plot

# Density plot comparing distributions of actual vs predicted prices
ggplot(density_long, aes(x = Value, fill = Type)) +
  geom_density(alpha = 0.5) +              # Smoothed density curves
  scale_fill_manual(values = c("Actual" = "red",
                               "Predicted" = "blue")) +
  theme_bw() +
  labs(
    title = "Density Plot: Actual vs Predicted Prices (Linear Regression)",
    x = "Price",
    y = "Density"
  )
#=========================EXTENDING MODEL: RANDOM FOREST====================
# Set seed for reproducibility
set.seed(1997)  # Fix seed to ensure consistent results

# Build a Random Forest model to predict car prices
rf_model <- randomForest(Price ~ Brand + Model + Year + Country + Condition +
                           Engine_Size + Horsepower + Torque + Weight + Top_Speed +
                           Acceleration_0_100 + Fuel_Type + Drivetrain + Transmission +
                           Fuel_Efficiency + CO2_Emissions + Mileage + Popularity +
                           Safety_Rating + Number_of_Owners + Market_Demand + Insurance_Cost +
                           Production_Units, data = test_data,
                         ntree = 500,  # Number of trees in the forest
                         mtry = 4,  # Number of variables randomly selected at each split
                         importance = TRUE  # Compute importance of variables
)

# Print the model summary
print(rf_model)

# View the importance of the variables used in the model
importance(rf_model)

# Plot the importance of variables
varImpPlot(rf_model, main = "Variable Importance in Random Forest")

# Make predictions on the testing data
test_data$Pred_RF <- predict(rf_model, newdata = test_data)

# Compute MAE, MSE and RMSE for Random Forest
mae_rf <- mean(abs(test_data$Price - test_data$Pred_RF))
mse_rf <- mean(((test_data$Price - test_data$Pred_RF)^2))  # MSE calculation
rmse_rf <- sqrt(mse_rf)  # RMSE calculation

# Compute R-squared (R²) for Random Forest model
ss_total <- sum((test_data$Price - mean(test_data$Price))^2)  # Total sum of squares
ss_res <- sum((test_data$Price - test_data$Pred_RF)^2)  # Residual sum of squares
r2_rf <- 1 - (ss_res / ss_total)  # R-squared

# Print the evaluation metrics
cat("MAE (Random Forest) =", round(mae_rf, 4), "\n")
cat("MSE (Random Forest) =", round(mse_rf, 4), "\n")
cat("RMSE (Random Forest) =", round(rmse_rf, 4), "\n")
cat("R^2 (Random Forest) =", round(r2_rf, 4), "\n")

# Plot the comparison between actual and predicted prices using Random Forest
ggplot(test_data, aes(x = Price, y = Pred_RF)) + 
  geom_point(alpha = 0.4, color = "darkblue") + 
  theme_minimal() + 
  labs(
    title = "Actual vs Predicted Price (Random Forest)", 
    x = "Actual Price", 
    y = "Predicted Price"
  )



