# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(vroom)
library(patchwork)
library(poissonreg)

## Penalized regression model ##

# Read in training data again to reset file
bike_training_data <- vroom('train.csv')

# Read in test data again to reset file
bike_test_data <- vroom('test.csv')

# Eliminate casual and registered columns from training data and 
#change count to log(count)
bike_training_data <- bike_training_data |>
  select(-casual, -registered) |>
  mutate(count = log(count))

# Create recipe for penalized regression model
penalized_recipe <- recipe(count ~ ., data=bike_training_data) |>
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
  fit(data=bike_training_data)

penalized_preds <- predict(penalized_workflow, new_data = bike_test_data) |>
  mutate(.pred = exp(.pred))


# Format the Predictions for Submission to Kaggle
kaggle_submission <- penalized_preds %>%
  bind_cols(., bike_test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./Penalized_Regression.csv", delim = ",")