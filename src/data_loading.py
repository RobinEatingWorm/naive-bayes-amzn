import pandas as pd

import re


def _load_bezos_tweets() -> pd.DataFrame:
    """Return a DataFrame of Tweets from @JeffBezos.
    Source: https://www.kaggle.com/datasets/thedevastator/jeff-bezos-tweets-and-social-media-interactions
    """
    return pd.read_csv("../data/JeffBezos.csv")


def _load_amzn_prices() -> pd.DataFrame:
    """Return a DataFrame of AMZN stock prices up to 2023-04-04.
    Source: https://www.nasdaq.com/market-activity/stocks/amzn/historical
    """
    return pd.read_csv("../data/HistoricalData_1680662815050.csv")


def _price_to_float(price: str) -> float:
    """Convert a price string to a floating point number."""
    return float(re.sub(r"\$", "", price))


def _remove_time(date: str) -> str:
    """Remove the time from a date string."""
    return re.sub(r" [0-2][0-9]:[0-5][0-9]:[0-5][0-9]\+00:00$", "", date)


def load_data(n: int) -> pd.DataFrame:
    """Return a DataFrame containing Tweets from @JeffBezos, the dates they were posted,
    and the relative price change of AMZN stocks after n business days.
    """
    bezos_tweets = _load_bezos_tweets()
    amzn_prices = _load_amzn_prices()

    # Compute the relative change of AMZN stocks
    final_price = amzn_prices["Close/Last"].apply(_price_to_float).shift(n - 1)
    initial_price = amzn_prices["Open"].apply(_price_to_float)
    amzn_prices["relative_change"] = (final_price - initial_price) / initial_price

    # Merge the two DataFrames
    bezos_tweets["date"] = pd.to_datetime(bezos_tweets["date"].apply(_remove_time), format="%Y-%m-%d")
    amzn_prices["date"] = pd.to_datetime(amzn_prices["Date"], format="%m/%d/%Y")
    return bezos_tweets.merge(amzn_prices, on="date")
