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

