import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def assign_labels(relative_change: float) -> int:
    """Return a numeric label for the given relative change in price.

    Return:
    * 0 for decrease (relative_change < 0)
    * 1 for increase or no change (relative_change >= 0)
    """
    return int(relative_change >= 0)


def vectorize(data: np.ndarray, max_features: int, tfidf: bool = False) -> tuple:
    """Vectorize the data by token counts. If tfidf is True, vectorize the data by TF-IDF
    counts instead.

    Return:
    * matrix: an n by d matrix of counts where n is the number of observations and d is
              the number of features (tokens) which cannot be greater than max_features
    * features: a vector with all feature (token) names
    """
    if tfidf:
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        vectorizer = CountVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(data).toarray()
    features = vectorizer.get_feature_names_out()
    return matrix, features


def naive_bayes(matrix: np.ndarray, labels: np.ndarray) -> MultinomialNB:
    """Fit a multinomial naive Bayes model using the matrix and labels."""
    nb_model = MultinomialNB()
    nb_model.fit(matrix, labels)
    return nb_model


def accuracy(nb_model: MultinomialNB, matrix: np.ndarray, labels: np.ndarray) -> float:
    """Return the accuracy of the multinomial naive Bayes model on the matrix and labels."""
    return accuracy_score(nb_model.predict(matrix), labels)


def highest_freq_tokens(label: int, k: int, matrix: np.ndarray, features: np.ndarray, labels: np.ndarray) -> list:
    """Return the top k tokens ranked by frequency out of all observations with the given label."""
    num_features = features.shape[0]
    num_obs = matrix.shape[0]

    # Calculate all token frequencies
    freq = np.zeros(num_features)
    for i in range(num_obs):
        if labels[i] == label:
            freq += matrix[i]

    # Get the top k tokens with the highest frequencies
    tokens = []
    index_min = None
    for i in range(num_features):
        if len(tokens) >= k:
            if freq[i] < tokens[index_min][1]:
                continue
            tokens.pop(index_min)
        tokens.append((features[i], freq[i]))
        tokens_freq = list(zip(*tokens))[1]
        index_min = tokens_freq.index(min(tokens_freq))
    return sorted(tokens, key=lambda t: t[1], reverse=True)
