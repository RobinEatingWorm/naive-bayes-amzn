from sklearn.model_selection import train_test_split

from data_loading import load_data
from naive_bayes import assign_labels, vectorize, naive_bayes, accuracy, highest_freq_tokens
from nlp import tokenize, remove_urls, to_lower, remove_stopwords, stem, untokenize


def main() -> None:
    # Load the data
    data = load_data(4)

    # Tokenize and process the observations
    data_obs = data["content"].apply(tokenize).apply(remove_urls).apply(to_lower) \
        .apply(remove_stopwords).apply(stem).apply(untokenize).to_numpy()

    # Assign labels to each observation
    data_y = data["relative_change"].apply(assign_labels).to_numpy()

    # Vectorize the data by TF-IDF counts
    data_X, data_features = vectorize(data_obs, 15, tfidf=True)

    # Split the data into training, validation, and testing datasets
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.2, random_state=363657441)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.25, random_state=987007865)

    # Fit a multinomial naive Bayes classifier with the training dataset
    nb_model = naive_bayes(train_X, train_y)

    # Get the accuracy of the three datasets
    print(f"Training Accuracy:   {accuracy(nb_model, train_X, train_y)}")
    print(f"Validation Accuracy: {accuracy(nb_model, valid_X, valid_y)}")
    print(f"Testing Accuracy:    {accuracy(nb_model, test_X, test_y)}")

    # Get the highest frequency tokens for each label (0 and 1)
    k = 5
    print(f"Highest Frequency Tokens (Label 0): {highest_freq_tokens(0, k, data_X, data_features, data_y)}")
    print(f"Highest Frequency Tokens (Label 1): {highest_freq_tokens(1, k, data_X, data_features, data_y)}")


if __name__ == '__main__':
    main()
