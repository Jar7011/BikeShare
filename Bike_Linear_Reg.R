# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(vroom)
library(patchwork)
library(poissonreg)

## Linear Regression ##

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

# Clean and tidy up test data to match training data format
bike_test_data$season <- factor(bike_test_data$season,
                                levels = c(1,2,3,4),
                                labels = c('Spring', 'Summer', 
                                           'Fall', 'Winter'))
bike_test_data$weather <- factor(bike_test_data$weather,
                                 levels = c(1,2,3,4))
bike_test_data <- bike_test_data |>
  mutate(weather = ifelse(weather == 4, 3, weather))

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


