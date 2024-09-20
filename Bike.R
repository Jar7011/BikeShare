# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(vroom)
library(patchwork)
library(poissonreg)

# Read in training data
bike_training_data <- vroom('train.csv')

# Read in test data
bike_test_data <- vroom('test.csv')

## Cleaning and organizing data ##

# Clean and tidy up training data
bike_training_data <- bike_training_data |>
  select(-casual, -registered)

bike_training_data$season <- factor(bike_training_data$season,
                                    levels = c(1,2,3,4),
                                    labels = c('Spring', 'Summer', 
                                               'Fall', 'Winter'))
bike_training_data$weather <- factor(bike_training_data$weather,
                                     levels = c(1,2,3,4))
bike_training_data <- bike_training_data |>
  mutate(weather = ifelse(weather == 4, 3, weather))

# Read in test data
bike_test_data <- vroom('test.csv')

# Clean and tidy up test data to match training data format
bike_test_data$season <- factor(bike_test_data$season,
                                    levels = c(1,2,3,4),
                                    labels = c('Spring', 'Summer', 
                                               'Fall', 'Winter'))
bike_test_data$weather <- factor(bike_test_data$weather,
                                     levels = c(1,2,3,4))
bike_test_data <- bike_test_data |>
  mutate(weather = ifelse(weather == 4, 3, weather))

# Some initial EDA plots
glimpse(bike_training_data)
plot_intro(bike_training_data)
plot_correlation(bike_training_data)
plot_missing(bike_training_data)

# Create four other EDA plots
weather_barplot <- ggplot(bike_training_data, aes(x=weather)) +
  geom_bar() +
  labs(x = 'Weather Condition',
       y = 'Number of Observations',
       title = 'Observations of Weather Conditions') +
  theme_minimal()

temp_atemp_corr <- ggplot(bike_training_data, aes(x=temp, y=atemp)) +
  geom_point() + 
  labs(x = 'Actual Temperature (°C)',
       y = "'Feels like' Temperature (°C)",
       title = 'Actual vs Perceived Temperature') + 
  theme_minimal()

working_day_plot <- ggplot(bike_training_data, aes(x=workingday)) +
  geom_bar() +
  labs(x = 'Working Day',
       y = 'Number of Observations',
       title = 'Using Bikes for Work Commute') + 
  theme_minimal()

temp_histogram <- ggplot(bike_training_data, aes(x=humidity)) +
  geom_histogram() +
  labs(x = 'Temperature (°C)',
       y = 'Number of Observations',
       title = 'Temperature Distribution') +
  theme_minimal()

# Put the four plots together
(weather_barplot + working_day_plot) / (temp_atemp_corr + temp_histogram)

## Linear Regression ##

# Set up linear regression model
linear_model <- linear_reg() |>
  set_engine('lm') |>
  set_mode('regression') |>
  fit(formula=log(count)~., data=bike_training_data)

# Make predictions
lm_predictions <- exp(predict(linear_model,
                              new_data = bike_test_data))

# Format the Predictions for Submission to Kaggle
kaggle_submission <- lm_predictions %>%
  bind_cols(., bike_test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./Linear_Predictions.csv", delim = ",")

## Poisson Regression ##

# Set up poisson regression model
poisson_model <- poisson_reg() |>
  set_engine('glm') |>
  set_mode('regression') |>
  fit(formula=count~., data=bike_training_data)

# Make predictions
pois_predictions <- predict(poisson_model,
                            new_data = bike_test_data)

# Format the Predictions for Submission to Kaggle
pois_kaggle_submission <- pois_predictions %>%
  bind_cols(., bike_test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = pois_kaggle_submission, file = "./Poisson_Predictions.csv", delim = ",")


## Linear model with recipe and workflow ##

# Read in training data again to reset file
bike_training_data_2 <- vroom('train.csv')

# Read in test data again to reset file
bike_test_data_2 <- vroom('test.csv')

# Eliminate casual and registered columns from training data and 
#change count to log(count)
bike_training_data_2 <- bike_training_data_2 |>
  select(-casual, -registered) |>
  mutate(count = log(count))

# Create recipe
bike_recipe <- recipe(count ~ ., data=bike_training_data_2) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_mutate(weather = factor(weather, levels = c(1,2,3))) |>
  step_time(datetime, features = 'hour') |>
  step_mutate(season = factor(season,
                              levels = c(1,2,3,4),
                              labels = c('Spring', 'Summer', 
                                         'Fall', 'Winter'))) |>
  step_rm(atemp)

# Define model
lin_model <- linear_reg() |>
  set_engine('lm') |>
  set_mode('regression')

# Combine into workflow
bike_workflow <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(lin_model) |>
  fit(data = bike_training_data_2)

# Get predictions
lin_preds <- predict(bike_workflow, new_data = bike_test_data_2) |>
  mutate(.pred = exp(.pred))

# Format the Predictions for Submission to Kaggle
kaggle_submission <- lin_preds %>%
  bind_cols(., bike_test_data_2) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./Linear_Predictions_Workflow.csv", delim = ",")

## Penalized regression model ##

# Create recipe for penalized regression model
penalized_recipe <- recipe(count ~ ., data=bike_training_data_2) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_mutate(weather = factor(weather, levels = c(1,2,3))) |>
  step_time(datetime, features = 'hour') |>
  step_date(datetime, features = 'dow') |>
  step_mutate(season = factor(season,
                              levels = c(1,2,3,4),
                              labels = c('Spring', 'Summer', 
                                         'Fall', 'Winter'))) |>
  step_rm(atemp, datetime) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# Create penalized regression model
penalized_model <- linear_reg(penalty = 0.01, mixture = 0.1) |>
  set_engine('glmnet')
penalized_workflow <- workflow() |>
  add_recipe(penalized_recipe) |>
  add_model(penalized_model) |>
  fit(data=bike_training_data_2)
penalized_preds <- predict(penalized_workflow, new_data = bike_test_data_2) |>
  mutate(.pred = exp(.pred))
  

# Format the Predictions for Submission to Kaggle
kaggle_submission <- penalized_preds %>%
  bind_cols(., bike_test_data_2) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./Penalized_Regression.csv", delim = ",")

