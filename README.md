# Elite Sports Cars Price Analysis Using Statistical Methods

This repository contains the final report for the group project in the Probability and Statistics course (MT2013) at Ho Chi Minh City University of Technology (HCMUT), Faculty of Computer Science and Engineering. The project explores the relationship between technical specifications, usage characteristics, and market factors of elite sports cars, focusing on predicting the **Price** metric through advanced statistical modeling.

## Project Overview

In this project, we analyze a dataset of high-performance sports cars to understand how key technical and market-related variables influence vehicle pricing. Using statistical techniques such as multiple linear regression, correlation analysis, nonparametric tests, and cross-validation, we build predictive models to forecast car prices. Key highlights include:

- **Data Preprocessing**: Handling missing values, detecting outliers via the IQR method, removing unnecessary variables, and standardizing categorical factors.
- **Exploratory Data Analysis (EDA)**: Visualizations like histograms, boxplots, scatter plots, density plots, and Pearson correlation matrices for continuous variables.
- **Inferential Statistics**: Hypothesis testing (Shapiro–Wilk, Levene), ANOVA / Kruskal–Wallis with Dunn post-hoc, and multicollinearity checks via VIF.
- **Modeling**: Building multiple linear regression models and a Random Forest regression model to predict Price, with feature importance analysis and residual diagnostics.
- **Evaluation**: Metrics such as MAE, MSE, RMSE, R², along with prediction intervals and scenario-based forecasting (e.g., changing engine-related parameters).
- **Insights**: Identifying key influencers like Engine Size, Horsepower, Torque, Mileage, and Condition on car prices, with practical implications for valuation and market analysis.

This work demonstrates the application of probability and statistics in automotive data analysis, providing actionable insights for buyers, sellers, and researchers.

## File Descriptions

### BTL_XSTK.pdf
The complete project report in Vietnamese, including introduction, methodology, data preprocessing, descriptive statistics, inferential analysis, model building, evaluation, and conclusions. It features detailed visualizations, statistical results, and discussions on limitations and future work.

### Elite Sports Cars in Data.csv
The main dataset used for analysis, containing technical specifications, usage information, and market-related attributes of elite sports cars, along with the target variable **Price**.

*Note*: The R script used for data analysis (`Code_BTL.R`) is included in the repository and documented below.

## Requirements

No specific software requirements for viewing the report—just a PDF reader like Adobe Acrobat or any modern web browser.

To replicate the analysis:
- **R** (version 4.x or higher recommended).
- Install the required R packages by running the following command in R:
  ```R
  install.packages(c(
    "stringr", "tidyr", "dplyr", "zoo", "Metrics", "caret", "MASS",
    "ggplot2", "reshape2", "mltools", "DescTools", "plotly", "car",
    "effectsize", "boot", "patchwork", "rstatix", "PMwR", "FSA", "questionr"
  ))
