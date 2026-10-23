################################################################################
# Flight arrival delay prediction - HarvardX PH125.9x Choose Your Own project
################################################################################

# Note on sources:
# - Dataset: "Flights" (Kaggle, mahoora00135), license CC0: Public Domain.
#   Original source: U.S. Bureau of Transportation Statistics.
# - A generative AI tool (Claude, by Anthropic) was used to help with small portions
#   of the exploratory analysis code, which were reviewed and adapted by the author.

# Packages (installed automatically if missing)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ranger)
library(lubridate)   # date functions (make_date, wday); loaded explicitly for tidyverse versions before 2.0

################################################################################
# Load the data
################################################################################
# Dataset: "Flights" (Kaggle, mahoora00135), license CC0: Public Domain
# Original source: U.S. Bureau of Transportation Statistics
# (all flights that departed from New York City airports in 2013)

# Relative paths to the data files
data_file <- "data/flights.csv"
zip_file  <- "data/flights.zip"

# If the csv file is not available, extract it from the zip file
if(!file.exists(data_file)) unzip(zip_file, exdir = "data")

# Reading the dataset
flights <- read_csv(data_file)

# First look at the data
dim(flights)               # number of rows and columns
glimpse(flights)           # column names, types and first values

# Number of missing values in each column
colSums(is.na(flights))

################################################################################
# Data cleaning and target variable
################################################################################

flights_clean <- flights %>%
  # Remove cancelled and diverted flights (where arrival delay is unknown)
  filter(!is.na(arr_delay)) %>%                      
  
  # Target variable: binary factor indicating if a flight arrived > 15 minutes late
  # This follows the official U.S. Bureau of Transportation Statistics standard
  mutate(delayed = factor(ifelse(arr_delay > 15, "yes", "no"),
                          levels = c("no", "yes"))) %>%   
  
  # Keep only the target variable and the 7 predictors known before departure
  select(delayed, month, day, sched_dep_time, carrier, origin, dest, distance)                         

# Verify the dimensions of the final cleaned dataset
dim(flights_clean)

# Calculate the baseline proportion of delayed flights in the dataset
mean(flights_clean$delayed == "yes")

################################################################################
# Train / final test split
################################################################################
# 10% of the flights are kept aside as a final test set, used only once at the end.
# The split is stratified by the target, so both sets keep ~23.7% delayed flights.

set.seed(42)
test_index <- createDataPartition(flights_clean$delayed, times = 1, p = 0.1, list = FALSE)

final_test <- flights_clean[test_index, ]
dev_set    <- flights_clean[-test_index, ]
rm(test_index)

# Check sizes and proportion of delayed flights in each set
dim(dev_set)
dim(final_test)
mean(dev_set$delayed == "yes")
mean(final_test$delayed == "yes")

################################################################################
# Data exploration (development set only)
################################################################################

# Delay rate by month
dev_set %>%
  group_by(month) %>%
  summarize(delay_rate = mean(delayed == "yes")) %>%
  ggplot(aes(x = factor(month), y = delay_rate)) +
  geom_col(fill = "steelblue") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Flight Delay Rate by Month", 
       x = "Month", 
       y = "Proportion of Delayed Flights")


# Delay rate by scheduled hour of departure
dev_set %>%
  # Feature engineering: extract the hour from sched_dep_time (e.g., 1530 -> 15)
  mutate(sched_hour = floor(sched_dep_time / 100)) %>%
  group_by(sched_hour) %>%
  summarize(delay_rate = mean(delayed == "yes")) %>%
  ggplot(aes(x = sched_hour, y = delay_rate)) +
  geom_col(fill = "steelblue") +
  scale_x_continuous(breaks = seq(0, 23, 2)) + # Shows labels every 2 hours
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Flight Delay Rate by Scheduled Departure Hour", 
       x = "Scheduled Departure Hour (24h format)", 
       y = "Proportion of Delayed Flights")

# Delay rate by origin airport
dev_set %>%
  group_by(origin) %>%
  summarize(delay_rate = mean(delayed == "yes"), n = n()) %>%
  ggplot(aes(x = reorder(origin, delay_rate), y = delay_rate)) +
  geom_col(fill = "steelblue") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Flight Delay Rate by Origin Airport", 
       x = "Origin Airport", 
       y = "Proportion of Delayed Flights")

# Carrier codes and full names (from the original data), used to identify the airlines
flights %>% distinct(carrier, name) %>% arrange(carrier)

# Delay rate by carrier
carrier_rates <- dev_set %>%
  group_by(carrier) %>%
  summarize(delay_rate = mean(delayed == "yes"), n = n())

carrier_rates %>% arrange(desc(delay_rate))   # table in the console

carrier_rates %>%
  ggplot(aes(x = reorder(carrier, delay_rate), y = delay_rate)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Flight Delay Rate by Airline Carrier", 
       x = "Airline Carrier", 
       y = "Proportion of Delayed Flights")

################################################################################
# Confounding Analysis: Share of each carrier's flights within each airport
################################################################################

# Calculating the internal market share of each airline per origin airport
airport_shares <- dev_set %>%
  count(origin, carrier) %>%
  group_by(origin) %>%
  mutate(share = n / sum(n)) %>%
  arrange(origin, desc(share)) %>%
  slice_head(n = 4)          # Keep the top 4 dominant carriers for each airport

# Display the resulting table in the console
print(airport_shares)

################################################################################
# Stratification: delay rate of the same carrier in each origin airport
################################################################################

# Evaluating the delay rate of each carrier across different airports
# to isolate the airport effect from the carrier effect (Confounding Section 18.2)
carrier_airport_strat <- dev_set %>%
  group_by(carrier, origin) %>%
  summarize(delay_rate = mean(delayed == "yes"), n = n(), .groups = "drop") %>%
  filter(n >= 1000) %>%                       # keep only reliable groups
  pivot_wider(id_cols = carrier, names_from = origin, values_from = delay_rate)

# Display the wide comparison table in the console
print(carrier_airport_strat, n = Inf)

################################################################################
# Delay rate by destination airport (dest)
################################################################################

# Calculating the delay rate and flight count for each unique destination
destination_rates <- dev_set %>%
  group_by(dest) %>%
  summarize(delay_rate = mean(delayed == "yes"), n = n())

# Display the top 10 most frequented destinations in the console
destination_rates %>% 
  arrange(desc(n)) %>% 
  head(10)

# Display the unique number of destinations in the dataset
nrow(destination_rates)

# Highest and lowest delay rates among destinations with at least 1,000 flights
destination_rates %>% filter(n >= 1000) %>% arrange(desc(delay_rate)) %>% head(5)
destination_rates %>% filter(n >= 1000) %>% arrange(delay_rate) %>% head(5)

# Number of rare destinations (fewer than 100 flights)
sum(destination_rates$n < 100)

################################################################################
# Delay rate by flight distance (grouped into bins)
################################################################################

# Group the distance (in miles) into bins
distance_bins_summary <- dev_set %>%
  mutate(distance_bin = cut(distance, breaks = c(0, 500, 1000, 1500, 2000, 3000, 5000))) %>%
  group_by(distance_bin) %>%
  summarize(delay_rate = mean(delayed == "yes"), n = n())

# Display the resulting frequency and proportion table in the console
print(distance_bins_summary)

# Routes in the longest distance bin (over 3,000 miles): carriers and destinations
dev_set %>% filter(distance > 3000) %>% count(carrier, origin, dest)

################################################################################
# Delay rate by Day of the Month (day)
################################################################################

# Step 1: Calculate delay rate and flight count for each day of the month
day_rates <- dev_set %>%
  group_by(day) %>%
  summarize(delay_rate = mean(delayed == "yes"), n = n())

# Step 2: Print the summary table in the console
print(day_rates, n = Inf)

# Step 3: Plot a line chart to easily identify any cyclical or monthly patterns
day_rates %>%
  ggplot(aes(x = day, y = delay_rate)) +
  geom_line(color = "steelblue", linewidth = 1) + # Updated syntax for ggplot2 4.0+
  geom_point(color = "darkblue", size = 2) +
  scale_x_continuous(breaks = seq(1, 31, 2)) + # Shows ticks for odd days (1, 3, 5...)
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Flight Delay Rate by Day of the Month", 
       x = "Day of the Month", 
       y = "Proportion of Delayed Flights")

################################################################################
# Delay rate by day of the week (weekday)
################################################################################
# Day of the week is a recurring (structural) pattern, unlike the day of the month,
# whose peaks reflect events on specific dates of 2013

weekday_rates <- dev_set %>%
  mutate(weekday = factor(wday(make_date(2013, month, day), week_start = 1),
                          levels = 1:7,
                          labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))) %>%
  group_by(weekday) %>%
  summarize(delay_rate = mean(delayed == "yes"), n = n())

print(weekday_rates)

weekday_rates %>%
  ggplot(aes(x = weekday, y = delay_rate)) +
  geom_col(fill = "steelblue") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Flight Delay Rate by Day of the Week", 
       x = "Day of the Week", 
       y = "Proportion of Delayed Flights")

################################################################################
# Feature engineering: new predictors, factors and rare levels
################################################################################

# Step 1: Create new structural predictors and convert categorical variables into factors
# within the development set.
# Note: Keep the creation of 'weekday' BEFORE converting 'month' to factor, 
# as make_date requires numeric month values.
dev_set <- dev_set %>%
  mutate(
    weekday    = factor(wday(make_date(2013, month, day), week_start = 1),
                        levels = 1:7,
                        labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")),
    sched_hour = floor(sched_dep_time / 100),
    month      = factor(month),
    origin     = factor(origin)
  ) %>%
  # Drop raw columns that were replaced (day -> weekday, sched_dep_time -> sched_hour)
  select(-day, -sched_dep_time)

# Step 2: Apply the identical transformations to final_test matching training factor levels
final_test <- final_test %>%
  mutate(
    weekday    = factor(wday(make_date(2013, month, day), week_start = 1),
                        levels = 1:7,
                        labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")),
    sched_hour = floor(sched_dep_time / 100),
    month      = factor(month, levels = levels(dev_set$month)),
    origin     = factor(origin, levels = levels(dev_set$origin))
  ) %>%
  # Drop raw columns that were replaced (day -> weekday, sched_dep_time -> sched_hour)
  select(-day, -sched_dep_time)

# Step 3: Identify rare destinations with less than 100 flights in the dev_set
rare_destinations <- dev_set %>%
  count(dest) %>%
  filter(n < 100) %>%
  pull(dest)

# Step 4: Apply the grouping logic to the dev_set
# Rare destinations are collapsed into "Other". All unique carriers are preserved.
dev_set <- dev_set %>%
  mutate(
    dest    = factor(ifelse(dest %in% rare_destinations, "Other", dest)),
    carrier = factor(carrier) # Documenting OO (26 flights) as a known limitation
  )

# Step 5: Apply identical factor structures to final_test without data leakage
# Carriers use exact training levels. Destinations convert rare/unobserved entries to "Other"
final_test <- final_test %>%
  mutate(
    carrier = factor(carrier, levels = levels(dev_set$carrier)),
    dest    = factor(ifelse(dest %in% levels(dev_set$dest), dest, "Other"),
                     levels = levels(dev_set$dest))
  )

# Clean up working memory objects
rm(rare_destinations)

# Checks: flights grouped as "Other", no new carriers in final_test,
# identical factor levels in both sets and no missing values
print("Flights grouped into the 'Other' destination:")
table(dev_set$dest == "Other")

setdiff(unique(final_test$carrier), unique(dev_set$carrier))
identical(levels(dev_set$dest), levels(final_test$dest))
identical(levels(dev_set$carrier), levels(final_test$carrier))
sum(is.na(final_test))

################################################################################
# Validation strategy (inside the development set only)
################################################################################
# dev_set is split into a training part and a validation part (stratified by the target).
# Models are trained on a sample of the training part, with cross-validation for tuning,
# and compared on the validation part. final_test is NOT used here.

set.seed(42)
valid_index <- createDataPartition(dev_set$delayed, times = 1, p = 0.2, list = FALSE)

valid_set <- dev_set[valid_index, ]
train_set <- dev_set[-valid_index, ]
rm(valid_index)

# Sample of the training part (stratified) to keep computing time manageable
set.seed(42)
sample_index <- createDataPartition(train_set$delayed, times = 1,
                                    p = 50000 / nrow(train_set), list = FALSE)
train_sample <- train_set[sample_index, ]
rm(sample_index)

# Check sizes and proportion of delayed flights
dim(train_set); dim(valid_set); dim(train_sample)
mean(valid_set$delayed == "yes"); mean(train_sample$delayed == "yes")

# Check that no category is empty in the training sample (OO has very few flights)
sum(table(train_sample$dest) == 0)
sum(table(train_sample$carrier) == 0)
table(train_sample$carrier)["OO"]

################################################################################
# Model 1: Baseline Naive Model (Majority Class Predictor) on Validation Set
################################################################################

# The naive model predicts the most frequent class ("no") for all observations
# calculated over the internal validation set length
naive_preds_valid <- factor(rep("no", nrow(valid_set)), levels = c("no", "yes"))

# Evaluate baseline performance on the internal validation set 
# Specifying positive = "yes" ensures we correctly measure delay detection
baseline_confusion_valid <- confusionMatrix(naive_preds_valid, 
                                            valid_set$delayed, 
                                            positive = "yes")
print(baseline_confusion_valid)

################################################################################
# Model 2: Logistic regression with cross-validated threshold optimization
################################################################################

# Same resampling scheme for all models: 5-fold cross-validation on train_sample
# savePredictions = "final" preserves out-of-fold predictions to optimize the threshold
# classProbs = TRUE forces caret to calculate and store the underlying probabilities
ctrl <- trainControl(method = "cv",
                     number = 5,
                     savePredictions = "final",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)   # reports ROC (AUC), sensitivity and specificity

set.seed(42)

# Train the logistic regression model on the 50,001 flights sample
model_glm <- train(delayed ~ month + carrier + origin + dest + distance + sched_hour + weekday, 
                   data = train_sample, 
                   method = "glm", 
                   family = "binomial",
                   metric = "ROC",
                   trControl = ctrl)

print(model_glm)

# Coefficients of the numeric predictors (sign and size discussed in the report)
coef(model_glm$finalModel)[c("sched_hour", "distance")]

################################################################################
# Threshold Optimization using Cross-Validation Predictions (Preventing Data Leakage)
################################################################################

# Extract the out-of-fold validation predictions and probabilities from the trained model
cv_predictions <- model_glm$pred

# Grid search: evaluate thresholds from 0.10 to 0.60 with increments of 0.01
threshold_grid <- seq(0.10, 0.60, by = 0.01)

# Reusable function: Balanced accuracy for a given threshold, 
# from predicted probabilities and true classes
evaluate_threshold <- function(t, probs, obs) {
  preds <- factor(ifelse(probs > t, "yes", "no"), levels = c("no", "yes"))
  cm <- confusionMatrix(preds, obs, positive = "yes")
  return(cm$byClass["Balanced Accuracy"])
}

# Apply the flexible function across the entire grid using sapply parameters
cv_balanced_accuracies <- sapply(threshold_grid, evaluate_threshold,
                                 probs = cv_predictions$yes, obs = cv_predictions$obs)

# Find the threshold that maximizes the Cross-Validated Balanced Accuracy
glm_threshold  <- threshold_grid[which.max(cv_balanced_accuracies)]
glm_cv_bal_acc <- max(cv_balanced_accuracies)

# Print tuning results in the console
cat("Optimal Threshold Found via CV (GLM):", glm_threshold, "\n")
cat("Maximum Balanced Accuracy Obtained in CV (GLM):", round(glm_cv_bal_acc, 5), "\n")

# Plot the optimization curve for the report
tibble(threshold = threshold_grid, balanced_accuracy = cv_balanced_accuracies) %>%
  ggplot(aes(x = threshold, y = balanced_accuracy)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = glm_threshold, linetype = "dashed", color = "darkred") +
  labs(title = "Balanced accuracy by threshold (logistic regression, CV)",
       x = "Threshold", y = "Balanced accuracy")

################################################################################
# Evaluate the Optimized Logistic Regression Model on the independent valid_set
################################################################################

# Predict probabilities on the validation set
glm_probs_valid <- predict(model_glm, newdata = valid_set, type = "prob")

# Apply the model-specific optimal threshold found via cross-validation
glm_preds_opt_valid <- factor(ifelse(glm_probs_valid$yes > glm_threshold, "yes", "no"), 
                              levels = c("no", "yes"))

# Evaluate the optimized performance on the validation set
glm_confusion_opt_valid <- confusionMatrix(glm_preds_opt_valid, 
                                           valid_set$delayed, 
                                           positive = "yes")
print(glm_confusion_opt_valid)

# For comparison: the same model with the default threshold of 0.5
glm_preds_05_valid <- factor(ifelse(glm_probs_valid$yes > 0.5, "yes", "no"),
                             levels = c("no", "yes"))
glm_confusion_05_valid <- confusionMatrix(glm_preds_05_valid,
                                          valid_set$delayed,
                                          positive = "yes")
print(glm_confusion_05_valid)

################################################################################
# Model 3: Decision Tree (CART) with ROC Tuning and Custom Grid Search
################################################################################

# Set seed for reproducibility before tuning the tree
set.seed(42)

# Custom grid for the complexity parameter (cp), from a fully grown tree (0) to a
# simpler one (0.0002); larger values were tested and produced trees with no splits
tree_grid <- data.frame(cp = c(0, 0.00002, 0.00005, 0.0001, 0.0002))

# Train the Decision Tree model using rpart on the 50,001 flights sample
# The model inherits 'ctrl' and selects the optimal cp based on the maximum ROC Curve (AUC)
model_tree <- train(delayed ~ month + carrier + origin + dest + distance + sched_hour + weekday, 
                    data = train_sample, 
                    method = "rpart",
                    metric = "ROC",
                    tuneGrid = tree_grid,
                    trControl = ctrl)

# Print tree results to view the cross-validated ROC scores across the cp grid
print(model_tree)

################################################################################
# Threshold Optimization using Cross-Validation Predictions (Preventing Data Leakage)
################################################################################

# Extract the out-of-fold validation predictions and probabilities from the optimal tree
tree_cv_predictions <- model_tree$pred

# Apply our reusable function across the grid using tree out-of-fold probabilities
tree_cv_balanced_accuracies <- sapply(threshold_grid, evaluate_threshold,
                                      probs = tree_cv_predictions$yes, obs = tree_cv_predictions$obs)

# Find the threshold that maximizes the Cross-Validated Balanced Accuracy for the Tree
tree_threshold  <- threshold_grid[which.max(tree_cv_balanced_accuracies)]
tree_cv_bal_acc <- max(tree_cv_balanced_accuracies)

cat("Optimal Threshold Found via CV (Tree):", tree_threshold, "\n")
cat("Maximum Balanced Accuracy Obtained in CV (Tree):", round(tree_cv_bal_acc, 5), "\n")

# Plot the tree optimization curve
tibble(threshold = threshold_grid, balanced_accuracy = tree_cv_balanced_accuracies) %>%
  ggplot(aes(x = threshold, y = balanced_accuracy)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = tree_threshold, linetype = "dashed", color = "darkred") +
  labs(title = "Balanced accuracy by threshold (decision tree, CV)",
       x = "Threshold", y = "Balanced accuracy")

################################################################################
# Evaluate the Optimized Decision Tree Model on the independent valid_set
################################################################################

# Predict probabilities on the independent validation set
tree_probs_valid <- predict(model_tree, newdata = valid_set, type = "prob")

# Apply the model-specific optimal threshold found via tree cross-validation
tree_preds_opt_valid <- factor(ifelse(tree_probs_valid$yes > tree_threshold, "yes", "no"), 
                               levels = c("no", "yes"))

# Evaluate the optimized decision tree performance on the validation set
tree_confusion_opt_valid <- confusionMatrix(tree_preds_opt_valid, 
                                            valid_set$delayed, 
                                            positive = "yes")
print(tree_confusion_opt_valid)

################################################################################
# Model 4: Random Forest (ranger) with ROC Tuning and Parameter Search
################################################################################

set.seed(42)

# Custom hyperparameter tuning grid for ranger (Optimized to prevent long processing times)
# mtry: number of variables randomly sampled at each split (scaled up for high dummy-variable count)
# min.node.size: minimal number of flights in each terminal node (fixed at 20 to avoid leaves
#                with very few flights, which would harm the estimated probabilities)
rf_grid <- expand.grid(mtry = c(5, 11, 20),
                       splitrule = "gini",
                       min.node.size = 20)

# Train the Random Forest ensemble model using ranger on the 50,001 flights sample
# num.trees = 200: keeps computing time manageable compared to the default 500
# importance = 'impurity': calculates Gini importance for variable ranking later in the report
model_rf <- train(delayed ~ month + carrier + origin + dest + distance + sched_hour + weekday, 
                  data = train_sample, 
                  method = "ranger",
                  metric = "ROC",
                  tuneGrid = rf_grid,
                  trControl = ctrl,
                  num.trees = 200,
                  importance = "impurity")

# Print random forest results to view the cross-validated tuning matrix
print(model_rf)

################################################################################
# Threshold Optimization using Cross-Validation Predictions (Preventing Data Leakage)
################################################################################

# Extract the out-of-fold validation predictions and probabilities from the optimal forest
rf_cv_predictions <- model_rf$pred

# Apply our reusable function across the grid using random forest out-of-fold probabilities
rf_cv_balanced_accuracies <- sapply(threshold_grid, evaluate_threshold,
                                    probs = rf_cv_predictions$yes, obs = rf_cv_predictions$obs)

# Find the threshold that maximizes the Cross-Validated Balanced Accuracy for the Forest
rf_threshold   <- threshold_grid[which.max(rf_cv_balanced_accuracies)]
rf_cv_bal_acc  <- max(rf_cv_balanced_accuracies)

cat("Optimal Threshold Found via CV (Random Forest):", rf_threshold, "\n")
cat("Maximum Balanced Accuracy Obtained in CV (Random Forest):", round(rf_cv_bal_acc, 5), "\n")

# Plot the random forest optimization curve
tibble(threshold = threshold_grid, balanced_accuracy = rf_cv_balanced_accuracies) %>%
  ggplot(aes(x = threshold, y = balanced_accuracy)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_vline(xintercept = rf_threshold, linetype = "dashed", color = "darkred") +
  labs(title = "Balanced accuracy by threshold (Random Forest, CV)",
       x = "Threshold", y = "Balanced accuracy")

################################################################################
# Evaluate the Optimized Random Forest Model on the independent valid_set
################################################################################

# Predict probabilities on the independent validation set
rf_probs_valid <- predict(model_rf, newdata = valid_set, type = "prob")

# Apply the model-specific optimal threshold found via forest cross-validation
rf_preds_opt_valid <- factor(ifelse(rf_probs_valid$yes > rf_threshold, "yes", "no"), 
                             levels = c("no", "yes"))

# Evaluate the optimized random forest performance on the validation set
rf_confusion_opt_valid <- confusionMatrix(rf_preds_opt_valid, 
                                          valid_set$delayed, 
                                          positive = "yes")
print(rf_confusion_opt_valid)

################################################################################
# Model comparison on the validation set
################################################################################

# Key metrics extracted from a confusion matrix
get_metrics <- function(cm) {
  tibble(`Balanced accuracy` = cm$byClass["Balanced Accuracy"],
         Sensitivity         = cm$byClass["Sensitivity"],
         Specificity         = cm$byClass["Specificity"],
         Kappa               = cm$overall["Kappa"],
         PPV                 = cm$byClass["Pos Pred Value"],
         NPV                 = cm$byClass["Neg Pred Value"])
}

model_comparison <- bind_rows(get_metrics(baseline_confusion_valid),
                              get_metrics(glm_confusion_opt_valid),
                              get_metrics(tree_confusion_opt_valid),
                              get_metrics(rf_confusion_opt_valid)) %>%
  mutate(Model      = c("Baseline", "Logistic regression", "Decision tree", "Random forest"),
         `AUC (CV)` = c(NA, max(model_glm$results$ROC), max(model_tree$results$ROC),
                        max(model_rf$results$ROC)),
         Threshold  = c(NA, glm_threshold, tree_threshold, rf_threshold),
         .before = 1)

print(model_comparison, width = Inf)

################################################################################
# Variable importance of the random forest (aggregated by original predictor)
################################################################################
# ranger reports impurity importance for each dummy column; columns are summed
# back into the 7 original predictors

rf_importance <- varImp(model_rf, scale = FALSE)$importance %>%
  rownames_to_column("column") %>%
  mutate(variable = case_when(str_starts(column, "month")   ~ "month",
                              str_starts(column, "carrier") ~ "carrier",
                              str_starts(column, "origin")  ~ "origin",
                              str_starts(column, "dest")    ~ "dest",
                              str_starts(column, "weekday") ~ "weekday",
                              TRUE ~ column)) %>%          # distance and sched_hour keep their names
  group_by(variable) %>%
  summarize(importance = sum(Overall)) %>%
  arrange(desc(importance))

print(rf_importance)

rf_importance %>%
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Random forest variable importance (impurity)",
       x = "Predictor", y = "Importance (sum of impurity decrease)")

################################################################################
# Final Model Training: Re-training the Champion Random Forest on 100% dev_set
################################################################################

# Configure a single execution control without resampling (method = "none")
# to train the final model on the entire development volume without cross-validation
final_ctrl <- trainControl(method = "none", 
                           classProbs = TRUE)

# Automatically extract the optimal hyperparameters from the cross-validation stage
final_rf_grid <- model_rf$bestTune

# Set seed for strict reproducibility before final ensemble generation
set.seed(42)

print("Training the final Random Forest model on the complete dev_set (294,611 flights)...")

# Train the single final forest over the complete 100% development dataset
model_rf_final <- train(delayed ~ month + carrier + origin + dest + distance + sched_hour + weekday, 
                        data = dev_set, 
                        method = "ranger",
                        tuneGrid = final_rf_grid,
                        trControl = final_ctrl,
                        num.trees = 200,
                        importance = "none") # variable importance is reported from model_rf (tuning stage)

print("Final model training completed successfully.")

################################################################################
# FINAL EVALUATION: One-time execution on the untouched final_test set
################################################################################

# Predict probabilities on the final test set using the champion ensemble
rf_probs_test <- predict(model_rf_final, newdata = final_test, type = "prob")

# Apply the model-specific optimal threshold found via forest cross-validation
rf_preds_test <- factor(ifelse(rf_probs_test$yes > rf_threshold, "yes", "no"), 
                        levels = c("no", "yes"))

# Evaluate the final performance metrics on the final test set
final_confusion_test <- confusionMatrix(rf_preds_test, 
                                        final_test$delayed, 
                                        positive = "yes")

print("================================================================================")
print("                       FINAL UNTOUCHED TEST SET RESULTS                         ")
print("================================================================================")
print(final_confusion_test)

################################################################################
# Save results for the report
################################################################################
# The report (Rmd) loads these small summary tables instead of re-running the analysis

report_data <- list(
  # Data overview
  n_raw         = nrow(flights),
  n_clean       = nrow(flights_clean),
  missing       = colSums(is.na(flights)),
  delay_rate    = mean(flights_clean$delayed == "yes"),
  n_dev         = nrow(dev_set),
  n_final_test  = nrow(final_test),
  n_train_set   = nrow(train_set),
  n_valid_set   = nrow(valid_set),
  n_train_sample = nrow(train_sample),
  
  # Exploration (development set)
  month_rates  = dev_set %>% group_by(month) %>%
    summarize(delay_rate = mean(delayed == "yes"), n = n()),
  hour_rates   = dev_set %>% group_by(sched_hour) %>%
    summarize(delay_rate = mean(delayed == "yes"), n = n()),
  origin_rates = dev_set %>% group_by(origin) %>%
    summarize(delay_rate = mean(delayed == "yes"), n = n()),
  carrier_names         = flights %>% distinct(carrier, name) %>% arrange(carrier),
  carrier_rates         = carrier_rates,
  airport_shares        = airport_shares,
  carrier_airport_strat = carrier_airport_strat,
  destination_rates     = destination_rates,
  distance_bins_summary = distance_bins_summary,
  day_rates             = day_rates,
  weekday_rates         = weekday_rates,
  
  # Modeling
  glm_coefs        = coef(model_glm$finalModel)[c("sched_hour", "distance")],
  glm_confusion_05 = glm_confusion_05_valid,
  tree_tuning      = model_tree$results,
  rf_tuning        = model_rf$results,
  threshold_curves = tibble(threshold = threshold_grid,
                            logistic  = cv_balanced_accuracies,
                            tree      = tree_cv_balanced_accuracies,
                            forest    = rf_cv_balanced_accuracies),
  model_comparison = model_comparison,
  rf_importance    = rf_importance,
  
  # Final evaluation
  final_confusion  = final_confusion_test
)

saveRDS(report_data, "aviation_results.rds")