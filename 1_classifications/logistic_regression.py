"""

SI 630 | WN 2020 | Ji Hwang (Henry) Kim

HW 1 Logistic Regression: Classify tweets with Logistic Regression method, given labeled training data

"""

### Imports
import csv, re
import numpy as np
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

### Functions
def tokenize(tweet):
    """
    - Input:
        - tweet (string)
    - Output:
        - (list of strings)
    - Description: Seperates words in input string by white space and returns the list of words
    """
    return tweet.split()

def compute_f1(true_pos, false_pos, true_neg, false_neg):
    """
    - Input:
        - true_pos (int)
        - false_pos (int)
        - true_neg (int)
        - false_neg (int)
    - Output:
        - f1_score (float)
    - Description: Computes the F1 score from known formula
    """

    # Compute precision and recall values
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    # Compute F1 score and return it
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def sigmoid(in_vec):
    """
    - Input:
        - in_vec (vector of floats)
    - Output:
        - (vector of floats)
    - Description: Computes the sigmoid function of the input vector and returns it
    """
    return 1 / (1 + np.exp(-in_vec))

class LogisticRegression():
    def __init__(self):
        pass

    def train(self, data_filename, label_filename):
        """
        - Input:
            - data_filename (string)
            - label_filename (string)
        - Update:
            - X (matrix of ints)
            - Y (matrix of ints)
        - Description: Given labeled training data, create matrices X and Y and save them to class variables
        """

        # Open the training data file and labeled data file
        training_tweets = open(data_filename, "r")
        training_labels = open(label_filename, "r")

        # Initialize the dictionary to map columns for each word and indices lists with data lists
        self.column_keys = defaultdict(int)
        X_data = []
        X_row = []
        X_col = []
        Y_data = []

        # For each tweet, do:
        for i, tweet in enumerate(training_tweets):

            # Add 1 to the first column (intercept coefficient)
            X_row.append(i)
            X_col.append(0)
            X_data.append(1)

            # Tokenize the tweet into words
            words = tokenize(tweet)

            # For each word in the words, do:
            for word in words:

                # If not in the column_keys dictionary, add to it
                if word not in self.column_keys.keys():
                    self.column_keys[word] = len(self.column_keys.keys()) + 1

            # For each counted words in the tweet, do:
            for j, k in Counter(words).items():

                # Add the row index and column index and add how many appears in X_data
                X_row.append(i)
                X_col.append(self.column_keys[j])
                X_data.append(k)

            # Get the label of the tweet and store in Y_data
            Y_data.append(int(next(training_labels).strip()))

        # Input the data as sparse matrix given by scipy library
        self.X = csr_matrix((X_data, (X_row, X_col)), shape=(i + 1, len(self.column_keys.keys()) + 1))
        self.Y = Y_data

        # Close the files
        training_tweets.close()
        training_labels.close()

    def log_likelihood(self, j):
        """
        - Input:
            - j (int)
        - Output:
            - (float)
        - Description: Computes the log likelihood given row index for X and Y
        """
        return np.sum(self.Y[j] * self.X[j, :].dot(self.B) - np.log(1 + np.exp(self.X[j, :].dot(self.B))))

    def compute_gradient(self, j):
        """
        - Input:
            - j (int)
        - Output:
            - (vector of floats)
        - Description: Computes the gradient of the log likelihood and returns it
        """
        return self.X[j, :].T.dot(self.Y[j] - self.Y_hat)

    def logistic_regression(self, learning_rate, num_step):
        """
        - Input:
            - learning_rate (float)
            - num_step (int)
        - Output:
            - loss (vector of floats)
        - Description: Computes the loss, given learning rate and number of steps for SGD
        """

        # Reset the class variables
        self.B = np.zeros(self.X.shape[1])
        # self.Y_hat = np.zeros(self.Y.shape)
        self.loss = []

        # For num_step, do:
        for i in range(num_step):

            # Random sample row from X
            j = np.random.randint(self.X.shape[0])

            # Calculate the loss and append it to the list
            self.loss.append(self.log_likelihood(j))

            # Compute Y_hat using sigmoid function
            self.Y_hat = np.round(sigmoid(self.X[j, :].dot(self.B)))

            # Update the weights
            self.B += learning_rate * self.compute_gradient(j)

        return self.loss

    def predict(self, X):
        """
        - Input:
            - X (matrix of floats)
        - Output:
            - (int)
        - Description: Predicts the value of X given B
        """
        return np.round(sigmoid(X.dot(self.B)))

def main():
    """
    - Input:
    - Output:
    - Description: Main function to be ran when file is ran
    """

    # Initialize and train Logistic Regression classifier
    LR_classifier = LogisticRegression()
    LR_classifier.train("X_train.txt", "y_train.txt")

    # Try with learning_rate = 5e-5, num_step = 1000
    plt.plot(LR_classifier.logistic_regression(5e-5, 1000))
    plt.xlabel("num_step")
    plt.ylabel("log likelihood")
    plt.title("Logistic Regression with learning_rate=5e-5, num_step=1000")
    plt.show()

    # Try with much larger learning_rate = 5e-2, num_step = 1000
    plt.plot(LR_classifier.logistic_regression(5e-2, 1000))
    plt.xlabel("num_step")
    plt.ylabel("log likelihood")
    plt.title("Logistic Regression with learning_rate=5e-2, num_step=1000")
    plt.show()

    # Try with much smaller learning_rate = 5e-8, num_step = 1000
    plt.plot(LR_classifier.logistic_regression(5e-8, 1000))
    plt.xlabel("num_step")
    plt.ylabel("log likelihood")
    plt.title("Logistic Regression with learning_rate=5e-8, num_step=1000")
    plt.show()

    # Open the testing data file and label files
    testing_tweets = open("X_dev.txt", "r")
    testing_labels = open("y_dev.txt", "r")

    # Initialize indices lists with data lists
    X_data = []
    X_row = []
    X_col = []
    Y_data = []

    # For each tweet, do:
    for i, tweet in enumerate(testing_tweets):

        # Add 1 to the first column (intercept coefficient)
        X_row.append(i)
        X_col.append(0)
        X_data.append(1)

        # Tokenize the tweet into words
        words = tokenize(tweet)

        # For each counted words in the tweet, do:
        for j, k in Counter(words).items():

            # If the word is in the column_key dictionary, do:
            if j in LR_classifier.column_keys.keys():

                # Add the row index and column index and add how many appears in X_data
                X_row.append(i)
                X_col.append(LR_classifier.column_keys[j])
                X_data.append(k)

        # Get the label of the tweet and store in Y_data
        Y_data.append(int(next(testing_labels).strip()))

    # Input the data as sparse matrix given by scipy library
    X_test = csr_matrix((X_data, (X_row, X_col)), shape=(i + 1, len(LR_classifier.column_keys.keys()) + 1))

    # Close the files
    testing_tweets.close()
    testing_labels.close()

    # Initialize values needed for F1 score
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0

    # Predict the labels from trained classifier
    plt.plot(LR_classifier.logistic_regression(1e-1, int(2e5)))
    plt.xlabel("num_step")
    plt.ylabel("log likelihood")
    plt.title("Logistic Regression with learning_rate=1e-1, num_step=2e5")
    plt.show()
    computed_labels = LR_classifier.predict(X_test)

    # For each tweet in testing data set, do:
    for i in range(len(computed_labels)):

        # Get the true label
        true_label = Y_data[i]

        # Compare true label and computed label and add to correct category
        if computed_labels[i] == 1:
            if true_label == computed_labels[i]:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if true_label == computed_labels[i]:
                true_neg += 1
            else:
                false_neg += 1

    # Compute the F1 score and append it to the list
    f1_score = compute_f1(true_pos, false_pos, true_neg, false_neg)
    print("F1 score for the best model: {}".format(f1_score))

    # Open the real testing data file
    testing_tweets = open("X_test.txt", "r")

    # Initialize indices lists with data lists
    X_data = []
    X_row = []
    X_col = []

    # For each tweet, do:
    for i, tweet in enumerate(testing_tweets):

        # Add 1 to the first column (intercept coefficient)
        X_row.append(i)
        X_col.append(0)
        X_data.append(1)

        # Tokenize the tweet into words
        words = tokenize(tweet)

        # For each counted words in the tweet, do:
        for j, k in Counter(words).items():

            # If the word is in the column_key dictionary, do:
            if j in LR_classifier.column_keys.keys():

                # Add the row index and column index and add how many appears in X_data
                X_row.append(i)
                X_col.append(LR_classifier.column_keys[j])
                X_data.append(k)

    # Input the data as sparse matrix given by scipy library
    X_test = csr_matrix((X_data, (X_row, X_col)), shape=(i + 1, len(LR_classifier.column_keys.keys()) + 1))

    # Close the files
    testing_tweets.close()

    # Compute the predictions
    computed_labels = LR_classifier.predict(X_test)

    # Open the csv file to write to, then do:
    with open("y_test_lr.csv", "w", newline="") as f:

        # Initialize write for the file
        w = csv.writer(f)

        # Write initial row specified in Kaggle
        w.writerow(["Id", "Category"])

        # For each tweet in testing data set, do:
        for i in range(len(computed_labels)):

            # Write to each row of the file
            w.writerow([i, int(computed_labels[i])])


# When this file is ran, run main() function
if __name__ == '__main__':
    main()