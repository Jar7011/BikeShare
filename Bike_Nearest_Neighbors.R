# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)

## K Nearest Neighbor ##

# Read in training data
bike_training_data <- vroom('train.csv')

# Read in test data
bike_test_data <- vroom('test.csv')

# Eliminate casual and registered columns from training data and 
# change count to log(count)
bike_training_data <- bike_training_data %>% 
  select(-casual, -registered) %>% 
  mutate(count = log(count))

# Create recipe for regression tree model
nearest_neighbor_recipe <- recipe(count ~ ., data=bike_training_data) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = c(1,2,3))) %>% 
  step_time(datetime, features = 'hour') %>% 
  step_date(datetime, features = 'dow') %>% 
  step_mutate(season = factor(season,
                              levels = c(1,2,3,4),
                              labels = c('Spring', 'Summer', 
                                         'Fall', 'Winter'))) %>% 
  step_rm(atemp, datetime)

# Create model
nearest_neighbor_model <- nearest_neighbor(neighbors = tune(),
                                           dist_power = tune()) %>% 
  set_engine('kknn') %>% 
  set_mode('regression')

# Create workflow
nearest_neighbor_wf <- workflow() %>% 
  add_recipe(nearest_neighbor_recipe) %>% 
  add_model(nearest_neighbor_model)

# Create grid of values to tune over
nearest_neighbor_grid_params <- grid_regular(neighbors(),
                                             dist_power(),
                                             levels = 10)

# Split the data for cross validation
folds <- vfold_cv(bike_training_data, v = 10, repeats = 1)

# Run the CV
cv_results <- nearest_neighbor_wf %>% 
  tune_grid(resamples = folds,
            grid = nearest_neighbor_grid_params,
            metrics = metric_set(rmse))

# Find the best params
best_params <- cv_results %>% 
  select_best(metric = 'rmse')

# Finalize workflow
final_wf <- nearest_neighbor_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = bike_training_data)

# Predict
nearest_neighbor_preds <- predict(final_wf, new_data = bike_test_data) %>% 
  mutate(.pred = exp(.pred))

# Format the Predictions for Submission to Kaggle
kaggle_submission <- nearest_neighbor_preds %>%
  bind_cols(., bike_test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./Nearest_Neighbors.csv", delim = ",")
