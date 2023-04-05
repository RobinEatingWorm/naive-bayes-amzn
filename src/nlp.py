from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re


def tokenize(obs: str) -> list:
    """Tokenize the observation."""
    return obs.split()


def remove_urls(obs: list) -> list:
    """Remove all URL tokens from a tokenized observation."""
    return [re.sub("^http.*$", "", token) for token in obs]


def to_lower(obs: list) -> list:
    """Convert all letters to lowercase for each token of the observation."""
    return [token.lower() for token in obs]


def remove_stopwords(obs: list) -> list:
    """Remove stopwords from a tokenized observation."""
    return [token for token in obs if token not in stopwords.words("english")]


def stem(obs: list) -> list:
    """Stem each token of the observation."""
    return [PorterStemmer().stem(token) for token in obs]


def lemmatize(obs: list) -> list:
    """Lemmatize each token of the observation."""
    return [WordNetLemmatizer().lemmatize(token) for token in obs]


def untokenize(obs: list) -> str:
    """Joins a list of tokens into a string separated by spaces."""
    return " ".join(obs)
