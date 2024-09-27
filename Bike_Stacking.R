# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(stacks)

## Stacked Model ##

# Read in training data
bike_training_data <- vroom('train.csv')

# Read in test data
bike_test_data <- vroom('test.csv')

# Eliminate casual and registered columns from training data and 
# change count to log(count)
bike_training_data <- bike_training_data %>% 
  select(-casual, -registered) %>% 
  mutate(count = log(count))

# Create recipe for model
model_recipe <- recipe(count ~ ., data=bike_training_data) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = c(1,2,3))) %>% 
  step_time(datetime, features = 'hour') %>% 
  step_date(datetime, features = 'dow') %>% 
  step_mutate(season = factor(season,
                              levels = c(1,2,3,4),
                              labels = c('Spring', 'Summer', 
                                         'Fall', 'Winter'))) %>% 
  step_rm(atemp, datetime) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# Split data for cv
folds <- vfold_cv(bike_training_data, v = 10, repeats = 1)

# Create a control grid
untuned_model <- control_stack_grid()

# Penalized regression model
penalized_reg_model <- linear_reg(penalty = tune(),
                                  mixture = tune()) %>% 
  set_engine('glmnet')

# Set workflow
penalized_reg_wf <- workflow() %>% 
  add_recipe(model_recipe) %>% 
  add_model(penalized_reg_model)

# Grid of values to tune over
penalized_reg_grid_params <- grid_regular(penalty(),
                                          mixture(),
                                          levels = 5)

# Run the CV
penalized_red_models <- penalized_reg_wf %>% 
  tune_grid(resamples = folds,
            grid = penalized_reg_grid_params,
            metrics = metric_set(rmse),
            control = untuned_model)

# Random forest model

