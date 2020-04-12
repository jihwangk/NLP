from collections import defaultdict
from random import random, randint
from glob import glob
from math import log
import argparse
import os
import time

from nltk.corpus import stopwords
from nltk.probability import FreqDist

from nltk.tokenize import TreebankWordTokenizer
kTOKENIZER = TreebankWordTokenizer()
kDOC_NORMALIZER = True

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.wrappers import LdaMallet

class DocScanner:
    """
    Creates a vocabulary after scanning a corpus.
    """

    def __init__(self, lang="english", min_length=3, cut_first=100):
        """
        Set the minimum length of words and which stopword list (by language) to
        use.
        """
        self._stop = set(stopwords.words(lang))
        self._min_length = min_length
        self.docs = []

        print(("Using stopwords: %s ... " % " ".join(list(self._stop)[:10])))

    def scan(self, words):
        """
        Add a list of words as observed.
        """
        doc = []
        for ii in [x.lower() for x in words if x.lower() not in self._stop \
                       and len(x) >= self._min_length]:
            doc.append(ii)
        self.docs.append(doc)

def report(result, filename="default", limit=25):
    """
    Create a human readable report of topic probabilities to a file.
    """
    topicsfile = open(filename + ".txt", 'w')
    for topic in result:
        topicsfile.write("------------\nTopic %i\n------------\n" % \
                  (topic[0]))

        word = 0
        words_list = topic[1].split("*")
        for i in range(len(words_list) - 1):
            first_ind = words_list[i + 1].find('"')
            second_ind = words_list[i + 1].find('"', first_ind + 1)
            topicsfile.write("%0.5f\t%s\n" % \
                         (float(words_list[i][-5:]),
                          words_list[i + 1][first_ind + 1:second_ind]))

            word += 1
            if word > limit:
                break
    topicsfile.close()

def tokenize_file(filename):
    contents = open(filename).read()
    for ii in kTOKENIZER.tokenize(contents):
        yield ii

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", help="Where we read the source documents",
                           type=str, default=".", required=False)
    argparser.add_argument("--language", help="The language we use",
                           type=str, default="english", required=False)
    argparser.add_argument("--output", help="Where we write results",
                           type=str, default="result", required=False)
    argparser.add_argument("--vocab_size", help="Size of vocabulary",
                           type=int, default=1000, required=False)
    argparser.add_argument("--num_topics", help="Number of topics",
                           type=int, default=5, required=False)
    argparser.add_argument("--num_iterations", help="Number of iterations",
                           type=int, default=100, required=False)
    args = argparser.parse_args()

    doc_scanner = DocScanner(args.language)

    # Create a list of the files
    search_path = "%s/*.txt" % args.doc_dir
    files = glob(search_path)
    assert len(files) > 0, "Did not find any input files in %s" % search_path

    # Create the vocabulary
    for ii in files:
        doc_scanner.scan(tokenize_file(ii))

    # Initialize the documents
    docs = doc_scanner.docs
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # start = time.time()
    # gensim_lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=args.num_topics, iterations=args.num_iterations)
    # time_took = time.time() - start
    # report(gensim_lda.print_topics(num_topics=10, num_words=50), filename="gensim", limit=50)
    # print(("Total time it took: %0.5f seconds" % (time_took)))

    mallet_file = "/home/jihwangk/Desktop/GitDir/Mallet/bin/mallet"
    # start = time.time()
    mallet_lda = LdaMallet(mallet_file, corpus=corpus, num_topics=args.num_topics, id2word=dictionary, iterations=args.num_iterations)
    # time_took = time.time() - start
    mallet_lda = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet_lda, iterations=args.num_iterations)
    report(mallet_lda.print_topics(num_topics=10, num_words=50), filename="mallet", limit=50)
    # print(("Total time it took: %0.5f seconds" % (time_took)))
