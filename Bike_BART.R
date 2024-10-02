# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)

## BART Model ##

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
bart_recipe <- recipe(count ~ ., data=bike_training_data) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = c(1,2,3))) %>% 
  step_time(datetime, features = 'hour') %>% 
  step_date(datetime, features = 'dow') %>% 
  step_date(datetime, features = 'year') %>% 
  step_mutate(season = factor(season,
                              levels = c(1,2,3,4),
                              labels = c('Spring', 'Summer', 
                                         'Fall', 'Winter'))) %>% 
  step_rm(atemp, datetime)

# Create random forest model
bart_model <- bart(trees = 1000) %>% 
  set_engine('dbarts') %>% 
  set_mode('regression')

# Set workflow
bart_wf <- workflow() %>% 
  add_recipe(bart_recipe) %>% 
  add_model(bart_model) %>% 
  fit(data = bike_training_data)

# Predict
bart_preds <- predict(bart_wf, new_data = bike_test_data) %>% 
  mutate(.pred = exp(.pred))

# Format the Predictions for Submission to Kaggle
kaggle_submission <- bart_preds %>%
  bind_cols(., bike_test_data) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and prediction variables
  rename(count = .pred) %>% # Rename to count (for submission to Kaggle)
  mutate(count = pmax(0, count)) %>% # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format to Kaggle

# Write out the file
vroom_write(x = kaggle_submission, file = "./BART.csv", delim = ",")

## Score: 0.37236
