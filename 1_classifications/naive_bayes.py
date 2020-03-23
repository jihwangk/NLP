"""

SI 630 | WN 2020 | Ji Hwang (Henry) Kim

HW 1 Naive Bayes: Classify tweets with Naive Bayes method, given labeled training data

"""

### Imports
import csv, re
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

### Functions
def standard_tokenize(tweet):
    """
    - Input:
        - tweet (string)
    - Output:
        - (list of strings)
    - Description: Seperates words in input string by white space and returns the list of words
    """
    return tweet.split()

class NaiveBayes():
    def __init__(self, tokenize_func):

        # Initialize the probability dictionaries
        self.prior_x = defaultdict(int)
        self.prior_y = defaultdict(int)
        self.likelihood_x = defaultdict(int)
        self.word_count = defaultdict(int)
        self.smoothing_alpha = 0.
        self.tweet_total = 0
        self.tokenize = tokenize_func

    def train(self, data_filename, label_filename, smoothing_alpha=0.):
        """
        - Input:
            - data_filename (string)
            - label_filename (string)
            - smoothing_alpha (float)
        - Update:
            - prior_x (dictionary of ints)
            - prior_y (dictionary of ints)
            - likelihood_x (dictionary of list of ints)
        - Description: Given labeled training data and smoothing coefficient alpha, compute the prior probability of P(X = x_i), P(Y = y_i),
            and likelihood P(X = x_i | Y = y_i). Y = 1 is aggressive and Y = 0 is non-aggressive.
        """

        # Set class variables
        self.smoothing_alpha = smoothing_alpha

        # Open the training data file and labeled data file
        training_tweets = open(data_filename, "r")
        training_labels = open(label_filename, "r")

        # For each tweet, do:
        for tweet in training_tweets:

            # Get the label of the tweet
            label = int(next(training_labels).strip())

            # Tokenize the tweet into words
            words = self.tokenize(tweet)

            # For each word in the words, do:
            for word in words:

                # Add count to prior_x
                self.prior_x[word] += 1

                # If the word has been seen before, add count to likelihood_x
                try:
                    self.likelihood_x[word][label] += 1

                # If it hasn't been seen yet, construct the structure and count it
                except TypeError:
                    self.likelihood_x[word] = [0, 0]
                    self.likelihood_x[word][label] = 1

            # Add count to prior_y
            self.prior_y[label] += 1

            # Add count to word count dictionary based on word labels
            self.word_count[label] += len(words)

        # For each word seen, do:
        for word in self.prior_x.keys():

            # Divide each entry in prior_x by total number of words seen
            self.prior_x[word] /= sum(self.word_count.values())

            # Divide each entry in likelihood_x by total number of each labels seen
            self.likelihood_x[word][0] = (self.likelihood_x[word][0] + self.smoothing_alpha) / (self.word_count[0] + len(self.prior_x.keys()) * self.smoothing_alpha)
            self.likelihood_x[word][1] = (self.likelihood_x[word][1] + self.smoothing_alpha) / (self.word_count[1] + len(self.prior_x.keys()) * self.smoothing_alpha)

        # Normalize the labels seen
        self.tweet_total = sum(self.prior_y.values())
        self.prior_y[0] /= self.tweet_total
        self.prior_y[1] /= self.tweet_total

        # Close the files
        training_tweets.close()
        training_labels.close()

    def classify(self, words):
        """
        - Input:
            - words (list of strings)
        - Output:
            - (int)
        - Description: Given the list of words, classify the list and return the label
        """

        # Initialize log likelihoods, P(X = words | Y = 0) and P(X = words | Y = 1)
        label_0_prob = np.log(self.prior_y[0])
        label_1_prob = np.log(self.prior_y[1])

        # For each words, do:
        for word in words:

            # if there is smoothing, do:
            if self.smoothing_alpha != 0.:

                # If the word was not seen with negative label, do:
                if (type(self.likelihood_x[word]) != list) or (self.likelihood_x[word][0] == 0):

                    # Compute the log likelihood with smoothing coefficient
                    label_0_prob += np.log(self.smoothing_alpha / (self.word_count[0] + len(self.prior_x.keys()) * self.smoothing_alpha))

                # If the word was seen, do:
                else:

                    # Compute the log likelihood
                    label_0_prob += np.log(self.likelihood_x[word][0])

                # If the word was not seen with positive label, do:
                if (type(self.likelihood_x[word]) != list) or (self.likelihood_x[word][1] == 0):

                    # Compute the log likelihood with smoothing coefficient
                    label_1_prob += np.log(self.smoothing_alpha / (self.word_count[1] + len(self.prior_x.keys()) * self.smoothing_alpha))

                # If the word was seen, do:
                else:

                    # Compute the log likelihood
                    label_1_prob += np.log(self.likelihood_x[word][1])

            # If there is no smoothing, do:
            else:

                # If the word was seen in both positive and negative labels, do:
                if (word in self.likelihood_x) and (self.likelihood_x[word][0] != 0) and (self.likelihood_x[word][1] != 0):

                    # Compute the log likelihood
                    label_0_prob += np.log(self.likelihood_x[word][0])
                    label_1_prob += np.log(self.likelihood_x[word][1])

        # Return the label of higher probability
        if label_0_prob == label_1_prob:
            return np.random.randint(2)
        elif label_0_prob > label_1_prob:
            return 0
        else:
            return 1

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

def submission(tokenize):
    """
    - Input:
        - tokenize (tokenize function of choice)
    - Output:
    - Description: Prints the values and plots the graphs needed for submission, using input tokenize function
    """

    # Initialize and train Naive Bayes classifier
    NB_classifier = NaiveBayes(tokenize)
    NB_classifier.train("X_train.txt", "y_train.txt", 0)

    # Open the testing data file and labeled data file
    testing_tweets = open("X_dev.txt", "r")
    testing_labels = open("y_dev.txt", "r")

    # Initialize values needed for F1 score
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0

    # For each tweet in testing data set, do:
    for tweet in testing_tweets:

        # Get the true label
        true_label = int(next(testing_labels).strip())

        # Tokenize the words and classify the tweet
        words = tokenize(tweet)
        computed_label = NB_classifier.classify(words)

        # Compare true label and computed label and add to correct category
        if computed_label == 1:
            if true_label == computed_label:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if true_label == computed_label:
                true_neg += 1
            else:
                false_neg += 1

    # Compute the F1 score and print it to the console
    f1_score = compute_f1(true_pos, false_pos, true_neg, false_neg)
    print("F1 Score if no smoothing was used is: {}".format(f1_score))

    # Close the files
    testing_tweets.close()
    testing_labels.close()

    # Initialize varying smoothing alpha values and F1 score list
    alphas = np.linspace(0, 1.5, 16)
    f1_scores = []

    # Initialize best alpha finder
    best_alpha = 0.
    best_f1 = 0.

    # For different smoothing alpha value, do:
    for alpha in alphas:

        # Initialize and train Naive Bayes classifier
        NB_classifier = NaiveBayes(tokenize)
        NB_classifier.train("X_train.txt", "y_train.txt", alpha)

        # Open the testing data file and labeled data file
        testing_tweets = open("X_dev.txt", "r")
        testing_labels = open("y_dev.txt", "r")

        # Initialize values needed for F1 score
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0

        # For each tweet in testing data set, do:
        for tweet in testing_tweets:

            # Get the true label
            true_label = int(next(testing_labels).strip())

            # Tokenize the words and classify the tweet
            words = tokenize(tweet)
            computed_label = NB_classifier.classify(words)

            # Compare true label and computed label and add to correct category
            if computed_label == 1:
                if true_label == computed_label:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if true_label == computed_label:
                    true_neg += 1
                else:
                    false_neg += 1

        # Compute the F1 score and append it to the list
        f1_score = compute_f1(true_pos, false_pos, true_neg, false_neg)
        f1_scores.append(f1_score)

        # Update best alpha value
        if f1_score > best_f1:
            best_f1 = f1_score
            best_alpha = alpha

        # Close the files
        testing_tweets.close()
        testing_labels.close()

    # Print the best alpha value of best model with the F1 score
    print("The best alpha value is: {}, where the F1 score is {}".format(best_alpha, best_f1))

    # Plot the graph using calculated F1 scores
    plt.plot(alphas, f1_scores)
    plt.title("Smoothing Alpha vs F1 score with Standard Tokens")
    plt.xlabel("alpha")
    plt.ylabel("F1 Score")
    plt.show()

    # Open the csv file to write to, then do:
    with open("y_test_nb.csv", "w", newline="") as f:

        # Initialize write for the file
        w = csv.writer(f)

        # Write initial row specified in Kaggle
        w.writerow(["Id", "Category"])

        # Initialize and train Naive Bayes classifier with the best model
        NB_classifier = NaiveBayes(tokenize)
        NB_classifier.train("X_train.txt", "y_train.txt", best_alpha)

        # Open the testing data file and labeled data file
        testing_tweets = open("X_test.txt", "r")

        # For each tweet in testing data set, do:
        for i, tweet in enumerate(testing_tweets):

            # Tokenize the words and classify the tweet
            words = tokenize(tweet)
            computed_label = NB_classifier.classify(words)

            # Write to each row of the file
            w.writerow([i, computed_label])

        # Close the file
        testing_tweets.close()


def better_tokenize(tweet):
    """
    - Input:
        - tweet (string)
    - Output:
        - (list of strings)
    - Description: Seperates words in input string by better tokenizing method and returns the list of words
    """

    # Change the smile emojis to word "em_smile"
    tweet = tweet.replace(":)", " em_smile ")
    tweet = tweet.replace("(:", " em_smile ")
    tweet = tweet.replace(":]", " em_smile ")
    tweet = tweet.replace("[:", " em_smile ")
    tweet = tweet.replace(":}", " em_smile ")
    tweet = tweet.replace("{:", " em_smile ")
    tweet = tweet.replace(":-)", " em_smile ")
    tweet = tweet.replace(":D", " em_smile ")
    tweet = tweet.replace("XD", " em_smile ")
    tweet = tweet.replace(":P", " em_smile ")
    tweet = tweet.replace("^^", " em_smile ")

    # Change the wink emojis to word "em_wink"
    tweet = tweet.replace(";)", " em_wink ")
    tweet = tweet.replace("(;", " em_wink ")
    tweet = tweet.replace(";]", " em_wink ")
    tweet = tweet.replace("[;", " em_wink ")
    tweet = tweet.replace(";}", " em_wink ")
    tweet = tweet.replace("{;", " em_wink ")
    tweet = tweet.replace(";-)", " em_wink ")
    tweet = tweet.replace(";D", " em_wink ")

    # Change the sad face emojis to word "em_sad"
    tweet = tweet.replace(":(", " em_sad ")
    tweet = tweet.replace("):", " em_sad ")
    tweet = tweet.replace(";(", " em_sad ")
    tweet = tweet.replace(":[", " em_sad ")
    tweet = tweet.replace(":-(", " em_sad ")
    tweet = tweet.replace("D:", " em_sad ")
    tweet = tweet.replace("D;", " em_sad ")
    tweet = tweet.replace("-_-", " em_sad ")
    tweet = tweet.replace("~_~", " em_sad ")

    # Change all love sign to word "em_love"
    tweet = re.sub('<3[^ ]+', ' em_love ', tweet)

    # Change all HTML code found to what they should mean
    tweet = tweet.replace("&amp;", " & ")
    tweet = tweet.replace("&gt", " > ")
    tweet = tweet.replace("&lt", " < ")
    tweet = tweet.replace("&;", "\'")
    tweet = tweet.replace("&quot", "\"")

    # Change elongated expression of ! or ? to single character
    tweet = re.sub('!![^ ]+', ' ! ', tweet)
    tweet = re.sub(r'!\?[^ ]+', ' ! ', tweet)
    tweet = re.sub(r'\?\?[^ ]+', ' ? ', tweet)
    tweet = re.sub(r'\?![^ ]+', ' ? ', tweet)

    # Decapitalize all letters
    for f in re.findall("[A-Z]", tweet):
        tweet = tweet.replace(f, f.lower())

    return tweet.split()


def main():
    """
    - Input:
    - Output:
    - Description: Main function to be ran when file is ran
    """

    submission(standard_tokenize)
    submission(better_tokenize)

    return

# When this file is ran, run main() function
if __name__ == '__main__':
    main()