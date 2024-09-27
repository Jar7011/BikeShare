# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)

## Random Forest Model ##

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
rand_forest_recipe <- recipe(count ~ ., data=bike_training_data) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = c(1,2,3))) %>% 
  step_time(datetime, features = 'hour') %>% 
  step_date(datetime, features = 'dow') %>% 
  step_mutate(season = factor(season,
                              levels = c(1,2,3,4),
                              labels = c('Spring', 'Summer', 
                                         'Fall', 'Winter'))) %>% 
  step_rm(atemp, datetime)

# Create random forest model
rand_forest_model <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 1000) %>% 
  set_engine('ranger') %>% 
  set_mode('regression')

# Set workflow
rand_forest_wf <- workflow() %>% 
  add_recipe(rand_forest_recipe) %>% 
  add_model(rand_forest_model)

# Grid of values to tune over
prep(rand_forest_recipe) %>% bake(bike_training_data) %>% ncol() # = 10
rand_forest_grid_params <- grid_regular(mtry(range = c(1, 10)),
                                        min_n(),
                                        levels = 5)

# Split data for cross validation
folds <- vfold_cv(bike_training_data, v = 10, repeats = 1)

# Run the cv
cv_results <- rand_forest_wf %>% 
  tune_grid(resamples = folds,
            grid = rand_forest_grid_params,
            metrics = metric_set(rmse))

# Find best tuning params
best_params <- cv_results %>% 
  select_best(metric = 'rmse')

# Finalize workflow and fit it
final_wf <- rand_forest_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = bike_training_data)

# Predict
rand_forest_preds <- predict(final_wf, new_data = bike_test_data) %>% 
  mutate(.pred = exp(.pred))

# Format the Predictions for Submission to Kaggle
kaggle_submission <- rand_forest_preds %>%
  bind_cols(., bike_test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./Random_Forest.csv", delim = ",")
