**NCAA March Madness - Python Predictor**

This project was done for Kaggle (https://www.kaggle.com/).  The code generates a CSV file with the teams with winning prediction indicator.

The Kaggle data set includes:

```
RegularSeasonResults.csv
NCAATourneyCompactResults.csv
RegularSeasonCompactResults.csv
Teams.csv
SampleSubmissionStage1.csv
```

You can download the files from https://www.kaggle.com/datasets/ncaa/ncaa-basketball and store the files in `data` folder.

The prediction is very Simple Logistic Regression Model right now with Elo Rating but I plan to add more to this.


Here are the things this code does:

1. Loads the Kaggle CSVs

2. Builds simple team-season features

3. Trains a logistic regression model

4. Adds Elo Rating

5. Generate submission file

6. Generate debug file with team names

