"""

SI 630 | WN 2020 | Ji Hwang (Henry) Kim

HW 2 Word Embeddings: Implement word2vec!

"""

### Imports
import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import math, random, scipy, numpy as np
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords
from numba import jit

#.................................................................................
#... global variables
#.................................................................................
random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10

vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from


#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................
def loadData(filename):

	# Get global variables
	global uniqueWords, wordcodes, wordcounts

	# Set override to True for debugging purposes, reading in completed objects
	override = True
	if override:
		fullrec = pickle.load(open("w2v_fullrec.p","rb"))
		wordcodes = pickle.load(open("w2v_wordcodes.p","rb"))
		uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
		wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
		return fullrec

	# Load in data from file
	handle = open(filename, "r", encoding="utf8")
	fullconts = handle.read().split("\n")

	# Apply simple tokenization with whitespaces and lowercases
	fullconts = " ".join(fullconts).lower()

	# Initialize variables
	fullrec = []
	origcounts = Counter()
	min_count = 50
	fullrec_filtered = []


	print ("Generating token stream...")

	# Get a set of stopwords
	stopword_set = set(stopwords.words("english"))

	# For each token in fullconts, do:
	for token in nltk.word_tokenize(str(fullconts)):

		# If it is a stopword, add <STOP> to keep track of its positions
		if token in stopword_set:
			fullrec.append("<STOP>")

		# Only if it is fully alphabetical, append to fullrec
		elif token.isalpha():
			fullrec.append(token)

	# Count the words in fullrec except <STOP>
	origcounts = Counter(fullrec)


	print ("Performing minimum thresholding..")

	# For each token in fullrec, do:
	for token in fullrec:

		# If the total count is less than min_count, replace with the word <UNK>
		if origcounts[token] >= min_count:
			fullrec_filtered.append(token)
		else:
			fullrec_filtered.append("<UNK>")

	# Count the words again in the fullrec_filtered
	wordcounts = Counter(fullrec_filtered)
	del wordcounts["<STOP>"]
	del wordcounts["<UNK>"]

	# Replace fullrec with filtered version
	fullrec = fullrec_filtered


	print ("Producing one-hot indicies")

	# Sort and store unique words in uniqueWords
	uniqueWords = sorted(wordcounts.keys())

	#For each unique word, do:
	for i, token in enumerate(uniqueWords):

		# Assign a place in word vector in sorted order
		wordcodes[token] = i

	# For each token in fullrec, do:
	for i, token in enumerate(fullrec):

		# If it is a unique word, find the index and replace the word with the index
		if token in uniqueWords:
			fullrec[i] = wordcodes[token]

	# Close input file handle
	handle.close()

	# Store these objects for later if indicated
	saving = True

	if saving:
		pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
		pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
		pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
		pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))

	# Output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes
	return fullrec


#.................................................................................
#... compute sigmoid value
#.................................................................................
@jit
def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................
def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):

	print ("Generating exponentiated count vectors")

	# Initialize variables
	max_exp_count = 0
	exp_count_array = []
	prob_dist = []
	cumulative_dict = {}
	table_size = 1e7

	# For each unique word, do:
	for unique in uniqueWords:

		# Compute the frequency to the power of exp_power and store it in same indices
		exp_count_array.append(wordcounts[unique] ** exp_power)

	# Compute the normalizing denominator
	max_exp_count = sum(exp_count_array)



	print ("Generating distribution")

	# Normalize the values in exp_count_array and save to prob_dist
	for i in range(len(exp_count_array)):
		prob_dist.append(exp_count_array[i] / max_exp_count)



	print ("Filling up sampling table")

	# For each word, do:
	for i, prob in enumerate(prob_dist):

		# Fill in the dictionary cumulative_dict with the number of one hot encoding to represent probability
		for j in range(round(table_size * prob)):
			cumulative_dict[len(cumulative_dict.keys())] = i

			# If the size of cumulative_dict becomes table_size, break
			if len(cumulative_dict.keys()) == table_size:
				break

	return cumulative_dict


#.................................................................................
#... generate a specific number of negative samples
#.................................................................................
def generateSamples(context_idx, num_samples):
	global samplingTable, uniqueWords, randcounter

	# Initialize variables
	results = []

	# Until we get number of samples needed, do:
	while len(results) != num_samples:

		# Get a random place in samplingTable
		new_idx = samplingTable[random.randint(0, len(samplingTable) - 1)]

		# Only add if not context index
		if new_idx == context_idx:
			continue
		else:
			results.append(new_idx)

	return results


@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, context_word, W1, W2, negative_indices):

	# Initialize variables
	h = np.array(W1[center_token])
	v_old = np.array(W2[context_word])
	negative_sum = 0
	negative_error = 0

	# Update W2 for positive context word
	W2[context_word] -= learning_rate * (sigmoid(np.dot(v_old, h)) - 1) * h

	# Update W2 for negative context words
	for negative_index in negative_indices:
		v_old_neg = np.array(W2[negative_index])
		W2[negative_index] -= learning_rate * sigmoid(np.dot(v_old_neg, h)) * h

		# Add to the negative sum and error term
		negative_sum += sigmoid(np.dot(v_old_neg, h)) * v_old_neg
		negative_error += np.log(sigmoid(np.negative(np.dot(W2[negative_index], h))))

	# Compute positive sum
	positive_sum = (sigmoid(np.dot(v_old, h)) - 1) * v_old

	# Update W1 for positive and negative context words
	W1[center_token] -= learning_rate * (positive_sum + negative_sum)

	# Compute the total error
	nll_new = -np.log(sigmoid(np.dot(W2[context_word], h))) - negative_error

	return nll_new


#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................
def trainer(curW1 = None, curW2=None):
	global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
	vocab_size = len(uniqueWords)           #... unique characters
	hidden_size = 100                       #... number of hidden neurons
	context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
	nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations


	# Determine how much of the full sequence we can use while still accommodating the context window
	start_point = int(math.fabs(min(context_window)))
	end_point = len(fullsequence)-(max(max(context_window),0))
	mapped_sequence = fullsequence

	# Initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output
	if curW1==None:
		np_randcounter += 1
		W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
		W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
	else:
		# Initialized from pre-loaded file
		W1 = curW1
		W2 = curW2

	# Set the training parameters
	epochs = 5
	num_samples = 2
	learning_rate = 0.05
	nll = 0
	iternum = 0

	# Begin actual training
	for j in range(0,epochs):
		print ("Epoch: ", j)
		prevmark = 0

		# For each epoch, redo the whole sequence...
		for i in range(start_point,end_point):

			if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
				print ("Progress: ", round(prevmark+0.1,1))
				prevmark += 0.1
			if iternum%10000==0:
				print ("Negative likelihood: ", nll)
				nll_results.append(nll)
				nll = 0

			# Determine which token is our current input. Remember that we're looping through mapped_sequence
			center_token = mapped_sequence[i]

			# Skip to next iteration if you found <UNK> or <STOP>
			if type(center_token) != int:
				continue

			iternum += 1
			# Propagate to each of the context outputs
			for k in range(0, len(context_window)):

				# Use context_window to find one-hot index of the current context token
				context_index = mapped_sequence[i + context_window[k]]

				# If this index is <UNK> or <STOP>, skip
				if type(context_index) != int:
					continue

				# Construct some negative samples
				negative_indices = generateSamples(context_index, num_samples)

				# Perform gradient descent on both weight matrices and keep track of the negative log-likelihood in variable nll
				nll += performDescent(num_samples, learning_rate, center_token, context_index, W1, W2, negative_indices)

	for nll_res in nll_results:
		print (nll_res)
	return [W1,W2]


#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................
def load_model():
	handle = open("saved_W1.data","rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("saved_W2.data","rb")
	W2 = np.load(handle)
	handle.close()
	return [W1,W2]


#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................
def save_model(W1,W2):
	handle = open("saved_W1.data","wb+")
	np.save(handle, W1, allow_pickle=False)
	handle.close()

	handle = open("saved_W2.data","wb+")
	np.save(handle, W2, allow_pickle=False)
	handle.close()


#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
	global word_embeddings, proj_embeddings
	if preload:
		[curW1, curW2] = load_model()
	else:
		curW1 = None
		curW2 = None
	[word_embeddings, proj_embeddings] = trainer(curW1,curW2)
	save_model(word_embeddings, proj_embeddings)


#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................
def morphology(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes

	# Initialize variables
	vectors = [word_seq[0], word_embeddings[wordcodes[word_seq[1]]]]
	vector_math = vectors[0] - vectors[1]
	outputs = []
	norm_values = []

	# For each unique word, do:
	for unique in uniqueWords:

		# Add the word vector of the unique word and get L2 norm
		norm_values.append(np.linalg.norm(vector_math + word_embeddings[wordcodes[unique]]))

	# Argsort the array so that indices with lowest norm values will be at the front, and get the top 10
	sorted_list = np.argsort(np.array(norm_values))[:10]

	# For each word, make the output into desired format
	for index in sorted_list:
		outputs.append({"word": uniqueWords[index], "dist": norm_values[index]})

	return outputs

#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................
def analogy(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes

	# Initialize variables
	vectors = [word_embeddings[wordcodes[word_seq[0]]],
			   word_embeddings[wordcodes[word_seq[1]]],
			   word_embeddings[wordcodes[word_seq[2]]]]
	vector_math = vectors[0] - vectors[1] - vectors[2]
	outputs = []
	norm_values = []

	# For each unique word, do:
	for unique in uniqueWords:

		# Add the word vector of the unique word and get L2 norm
		norm_values.append(np.linalg.norm(vector_math + word_embeddings[wordcodes[unique]]))

	# Argsort the array so that indices with lowest norm values will be at the front, and get the top 10
	sorted_list = np.argsort(np.array(norm_values))[:10]

	# For each word, make the output into desired format
	for index in sorted_list:
		outputs.append({"word": uniqueWords[index], "dist": norm_values[index]})

	return outputs


#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................
def prediction(word_1, word_2):
	global word_embeddings, uniqueWords

	# Get the word vectors
	vec_1 = word_embeddings[uniqueWords.index(word_1)]
	vec_2 = word_embeddings[uniqueWords.index(word_2)]

	# Compute the distance between vectors like quaternions, with 1 - cos(vec_1, vec_2)
	return 1 - cosine(vec_1, vec_2)

def get_neighbors(target_word):
	global uniqueWords, wordcodes

	# Initialize variables
	outputs = []
	similarities = []

	# For each unique word, do:
	for unique in uniqueWords:

		# Compute the distance between target word and unique word and append it to similarity list
		similarities.append(prediction(target_word, unique))

	# Argsort the array so that indices with highest similarities will be at the end, and get the top 10
	sorted_list = np.argsort(np.array(similarities))[-11:-1]

	# For each word, make the output into desired format
	for index in sorted_list:
		outputs.append({"word": uniqueWords[index], "score": similarities[index]})

	return outputs


#.................................................................................
#... Main Function
#.................................................................................
if __name__ == '__main__':

	# If filename is specified, do:
	if len(sys.argv) >= 2:

		# Save filename to variable
		filename = sys.argv[1]

		# Load the data from file
		fullsequence= loadData(filename)
		print ("Full sequence loaded...")
		print("Unique Words: \n {}".format(uniqueWords))
		print("Unique Words Total: {}".format(len(uniqueWords)))

		# Generate the negative sampling table
		print("Preparing negative sampling table")
		samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)

		# Train the vectors
		#train_vectors(preload=False)
		[word_embeddings, proj_embeddings] = load_model()

		# If filename is specified for intrinsic matrix, do:
		if len(sys.argv) == 3:

			print("Computing similarities for given pairs")

			# Open the files
			pairs = open(sys.argv[2], "r")
			output = open("output.csv", "w")

			# Initialize reader and writer
			reader = csv.reader(pairs)
			writer = csv.writer(output)

			# For each line in pairs, do:
			for i, line in enumerate(reader):

				# For the first line, follow the format
				if i == 0:
					writer.writerow(["id", "sim"])

				# For other lines, do:
				else:
					writer.writerow([i - 1, prediction(line[1], line[2])])

			# Close the files
			pairs.close()
			output.close()

		# Pick ten words you choose
		targets = ["good", "bad", "food", "apple",'tasteful','unbelievably','uncle','tool','think']

		# Make predictions on the words
		for targ in targets:
			print("Target: ", targ)
			bestpreds = (get_neighbors(targ))
			for pred in bestpreds:
				print (pred["word"],":",pred["score"])
			print ("\n")

		# Try Analogy Task: A is to B as C is to ?
		print("Apple is to fruit as Banana is to:\n {}".format(analogy(["apple", "fruit", "banana"])))

		# Try Morphological Task: Input is averages of vector combinations that use some morphological change
		s_suffix = []
		word_pairs = [["bananas", "banana"], ["apples", "apple"],["values", "value"]]
		for pair in word_pairs:
			s_suffix.append(word_embeddings[wordcodes[pair[0]]] - word_embeddings[wordcodes[pair[1]]])
		s_suffix = np.mean(s_suffix, axis=0)

		# Compare the results of using word_embeddings vs proj_embeddings
		print("Getting rid of -s from apples give us:]\n {}".format(morphology([s_suffix, "apples"])))
		print("Getting rid of -s from pears give us:]\n {}".format(morphology([s_suffix, "pears"])))

	# If filename is not provided, log error and exit
	else:
		print ("Please provide a valid input filename")
		sys.exit()


