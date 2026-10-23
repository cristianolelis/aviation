# Predicting Flight Arrival Delays from New York City Airports

HarvardX PH125.9x Data Science: Capstone – Choose Your Own Project

Author: Cristiano Camilo de Araujo Lelis

## Goal

Predict whether a flight departing from the three New York City airports (EWR, JFK and LGA) in 2013 will arrive more than 15 minutes late, using only information known before departure (month, day of the week, scheduled hour, carrier, origin, destination and distance). Four models were compared (a majority-class baseline, logistic regression, a decision tree and a random forest), with balanced accuracy as the main metric. The final random forest reached a balanced accuracy of 0.652 on the final test set.

## Files

| File | Description |
|---|---|
| `aviation.R` | R script with the full analysis: data loading, cleaning, exploration, feature engineering, modeling and final evaluation |
| `aviation_report.Rmd` | Report in R Markdown |
| `aviation_report.pdf` | Report knitted from the Rmd file |
| `aviation_results.rds` | Summary results saved by `aviation.R` and read by the Rmd file |
| `data/flights.zip` | Dataset (zipped `flights.csv`) |

## Data

"Flights" dataset by mahoora00135 on Kaggle (https://www.kaggle.com/datasets/mahoora00135/flights), license CC0: Public Domain. The original source of the flight records is the U.S. Bureau of Transportation Statistics. The same records are distributed in the `nycflights13` R package.

## How to run

1. Download or clone this repository, keeping the `data` folder next to `aviation.R`.
2. Open `aviation.R` in RStudio and set the working directory to the script's folder (Session > Set Working Directory > To Source File Location).
3. Run the whole script. Missing packages are installed automatically, and `data/flights.zip` is unzipped automatically if `data/flights.csv` is not found. The full run takes about 20 to 30 minutes on a conventional computer, mostly for the random forest.
4. To rebuild the report, knit `aviation_report.Rmd` (a LaTeX distribution such as TinyTeX is required for the PDF). The Rmd file reads `aviation_results.rds`; if this file is missing, it runs `aviation.R` first.
