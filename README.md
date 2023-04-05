# Predicting Amazon Stock Price Changes

This analysis seeks to determine how much predictive power Jeff Bezos's Tweets have on Amazon stock prices.
The primary techniques used were NLP tokenization and processing of Tweet text, followed by fitting a naive Bayes model to predict whether a Tweet would be associated with an increase or decrease in stock price after 4 business days.

## Files

This repository contains the following files.

* `data/HistoricalData_1680662815050.csv`: [AMZN Historical Data](https://www.nasdaq.com/market-activity/stocks/amzn/historical) from NASDAQ. The data was downloaded on August 24, 2023.
* `data/JeffBezos.csv`: [Jeff Bezos Tweets and Social Media Interactions](https://www.kaggle.com/datasets/thedevastator/jeff-bezos-tweets-and-social-media-interactions) from Kaggle.
* `src/data_loading.py`: Helper functions to load the two datasets and merge them together by date.
* `src/main.py`: The main analysis performed on the data.
* `src/naive_bayes.py`: Helper functions using the package `sklearn` to create and evaluate a naive Bayes model.
* `src/nlp.py`: Helper functions using the packages `re` and `nltk` to tokenize and process text.

## Results

### Accuracy

Training, validation, and testing datasets were created by splitting the original dataset in a 60:20:20 ratio. Model accuracy for each dataset is listed below, rounded to 3 decimal places.

* Training Accuracy: 0.583
* Validation Accuracy: 0.596
* Testing Accuracy: 0.577

Admittedly, the model's accuracy is quite low. This makes sense in context, however, because it is unreasonable to expect Jeff Bezos's Tweets to be the only factor affecting Amazon stock prices.

### Most Probable Tokens

Term frequency–inverse document frequency (TF-IDF) was used to determine the probability of tokens appearing for Tweets in each class.
For Tweets associated with a decrease in Amazon stock price, the 5 tokens with the highest probability to appear were 'team', 'blueorigin', 'gradatimferocit', 'today', and 'thank'.
For Tweets associated with an increase (or no change) in Amazon stock price, the 5 tokens with the highest probability to appear were 'thank', 'amazon', 'work', 'gradatimferocit', and 'blueorigin'.

This indicates that the model's low accuracy may also be due to how Tweets in both classes tend to be similar to each other, as 3 tokens are shared between the two classes' top 5 highest probability tokens.
Realistically, we would not expect the content of Tweets from Jeff Bezos to vary wildly in content.
