library(tidyverse)
library(tidymodels)
library(DataExplorer)
library(vroom)
library(patchwork)

bike_training_data <- vroom('train.csv')
glimpse(bike_training_data)
plot_intro(bike_training_data)
plot_correlation(bike_training_data)
plot_missing(bike_training_data)

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

(weather_barplot + working_day_plot) / (temp_atemp_corr + temp_histogram)


# Set up linear regression model
# linear_model <- linear_reg() |>
#   set_engine('lm') |>
#   set_mode('regression') |>
#   fit(formula)