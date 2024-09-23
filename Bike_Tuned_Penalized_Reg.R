# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(vroom)
library(patchwork)
library(poissonreg)

## Tuned Penalized Model ##

# Read in training data
bike_training_data <- vroom('train.csv')

# Read in test data
bike_test_data <- vroom('test.csv')

# Eliminate casual and registered columns from training data and 
#change count to log(count)
bike_training_data <- bike_training_data %>% 
  select(-casual, -registered) %>% 
  mutate(count = log(count))

# Create recipe for penalized regression model
tuned_penalized_recipe <- recipe(count ~ ., data=bike_training_data) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = c(1,2,3))) %>% 
  step_time(datetime, features = 'hour') %>% 
  step_date(datetime, features = 'dow') %>% 
  step_mutate(season = factor(season,
                              levels = c(1,2,3,4),
                              labels = c('Spring', 'Summer', 
                                         'Fall', 'Winter'))) %>% 
  step_rm(atemp, datetime) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Tuned penalized regression model
tuned_penalized_model <- linear_reg(penalty = tune(),
                                    mixture = tune()) %>% 
  set_engine('glmnet')

# Set Workflow
tuned_penalized_wf <- workflow() %>% 
  add_recipe(tuned_penalized_recipe) %>% 
  add_model(tuned_penalized_model)

# Grid of values to tune over
grid_tuning_params <- grid_regular(penalty(),
                                   mixture(),
                                   levels = 5)

# Split data for cross validation (CV)
folds <- vfold_cv(bike_training_data, v = 10, repeats = 1)

# Run the CV
cv_results <- tuned_penalized_wf %>%
  tune_grid(resamples = folds,
            grid = grid_tuning_params,
            metrics = metric_set(rmse, mae, rsq))

# Plot results
collect_metrics(cv_results) %>% 
  filter(.metric == 'rmse') %>% 
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) + 
  geom_line()

# Find best tuning params
best_params <- cv_results %>% 
  select_best(metric = 'rmse')

# Finalize workflow and fit it
final_wf <- tuned_penalized_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = bike_training_data)

# Predict
tuned_penalized_preds <- predict(final_wf, new_data = bike_test_data) %>% 
  mutate(.pred = exp(.pred))

# Format the Predictions for Submission to Kaggle
kaggle_submission <- tuned_penalized_preds %>%
  bind_cols(., bike_test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./Tuned_Penalized_Regression.csv", delim = ",")