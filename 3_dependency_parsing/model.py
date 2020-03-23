import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class ParserModel(nn.Module):


    def __init__(self, config, word_embeddings=None, pos_embeddings=None, dep_embeddings=None):
        super(ParserModel, self).__init__()

        self.config = config

        # These are the hyper-parameters for choosing how many embeddings to encode in the input layer.
        n_w = config.word_features_types # 18
        n_p = config.pos_features_types # 18
        n_d = config.dep_features_types # 12

        # Copy the Embedding data that we'll be using in the model.
        self.word_embeddings = word_embeddings
        self.pos_embeddings = pos_embeddings
        self.dep_embeddings = dep_embeddings

        # Create the first layer of the network that transform the input data to the hidden layer raw outputs.
        self.first_layer = nn.Linear((n_w + n_p + n_d) * config.embedding_dim + 1, config.l1_hidden_size)

        # Create a dropout layer here to randomly zero out a percentage of the activations
        self.dropout = nn.Dropout(config.keep_prob)

        # Create the output layer that maps the activation of the hidden layer to the output classes
        self.second_layer = nn.Linear(config.l1_hidden_size, config.num_classes)

        # Initialize the weights of both layers
        self.init_weights()

    def init_weights(self):
        # initialize each layer's weights to be uniformly distributed within this
        # range of +/-initrange.  This initialization ensures the weights have something to
        # start with for computing gradient descent and generally leads to
        # faster convergence.
        initrange = 0.1


    def lookup_embeddings(self, word_indices, pos_indices, dep_indices, keep_pos = 1):

        # Based on the IDs, look up the embeddings for each thing we need.  Note
        # that the indices are a list of which embeddings need to be returned.
        w_embeddings = self.word_embeddings(word_indices)
        p_embeddings = self.pos_embeddings(pos_indices)
        d_embeddings = self.dep_embeddings(dep_indices)

        return w_embeddings, p_embeddings, d_embeddings

    def forward(self, word_indices, pos_indices, dep_indices):
        """
        Computes the next transition step (shift, reduce-left, reduce-right)
        based on the current state of the input.


        The indices here represent the words/pos/dependencies in the current
        context, which we'll need to turn into vectors.
        """

        # Look up the embeddings for this prediction
        w_embeddings, p_embeddings, d_embeddings = self.lookup_embeddings(word_indices, pos_indices, dep_indices)

        # Flatten each of the embeddings into a single dimension
        batch_size = w_embeddings.shape[0]
        w_embeddings = w_embeddings.view(batch_size, -1)
        p_embeddings = p_embeddings.view(batch_size, -1)
        d_embeddings = d_embeddings.view(batch_size, -1)

        # Compute the raw hidden layer activations from the concatentated input embeddings
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        hidden_activations = self.first_layer(torch.cat([w_embeddings, p_embeddings, d_embeddings, torch.ones(batch_size, 1).to(device)], dim=1).to(device))

        # Compute the cubic activation function here.
        cubic_activation = torch.pow(hidden_activations, 3)

        # Now do dropout for final activations of the first hidden layer
        dropout = self.dropout(cubic_activation)

        # Multiply the activation of the first hidden layer by the weights of the second hidden layer
        # and pass that through a ReLU non-linearity for the final output activations
        output = F.relu(self.second_layer(dropout))

        return output


class MultiLayer_ParserModel(nn.Module):


    def __init__(self, config, word_embeddings=None, pos_embeddings=None, dep_embeddings=None, layer_num=2):
        super(MultiLayer_ParserModel, self).__init__()

        self.config = config

        # These are the hyper-parameters for choosing how many embeddings to encode in the input layer.
        n_w = config.word_features_types # 18
        n_p = config.pos_features_types # 18
        n_d = config.dep_features_types # 12

        # Copy the Embedding data that we'll be using in the model.
        self.word_embeddings = word_embeddings
        self.pos_embeddings = pos_embeddings
        self.dep_embeddings = dep_embeddings

        # Make architecture of [linear - batchnorm - tanh] x (layer_num - 1) - linear - relu
        begin_list = [("lin1", nn.Linear((n_w + n_p + n_d) * config.embedding_dim + 1, config.l1_hidden_size)),
                      ("batch1", nn.BatchNorm1d(config.l1_hidden_size)),
                      ("tanh1", nn.Tanh())]
        middle_list = []
        end_list = [("lin{}".format(layer_num), nn.Linear(config.l1_hidden_size, config.num_classes)),
                    ("relu{}".format(layer_num), nn.ReLU())]

        for i in range(2, layer_num):
            middle_list.append(("lin{}".format(i), nn.Linear(config.l1_hidden_size, config.l1_hidden_size)))
            middle_list.append(("batch{}".format(i), nn.BatchNorm1d(config.l1_hidden_size)))
            middle_list.append(("tanh{}".format(i), nn.Tanh()))

        total_list = begin_list + middle_list + end_list
        # print("What is total_list? {}".format(total_list))
        self.sequence = nn.Sequential(OrderedDict(total_list))

        # Initialize the weights of both layers
        self.init_weights()

    def init_weights(self):
        # initialize each layer's weights to be uniformly distributed within this
        # range of +/-initrange.  This initialization ensures the weights have something to
        # start with for computing gradient descent and generally leads to
        # faster convergence.
        initrange = 0.1


    def lookup_embeddings(self, word_indices, pos_indices, dep_indices, keep_pos = 1):

        # Based on the IDs, look up the embeddings for each thing we need.  Note
        # that the indices are a list of which embeddings need to be returned.
        w_embeddings = self.word_embeddings(word_indices)
        p_embeddings = self.pos_embeddings(pos_indices)
        d_embeddings = self.dep_embeddings(dep_indices)

        return w_embeddings, p_embeddings, d_embeddings

    def forward(self, word_indices, pos_indices, dep_indices):
        """
        Computes the next transition step (shift, reduce-left, reduce-right)
        based on the current state of the input.


        The indices here represent the words/pos/dependencies in the current
        context, which we'll need to turn into vectors.
        """

        # Look up the embeddings for this prediction
        w_embeddings, p_embeddings, d_embeddings = self.lookup_embeddings(word_indices, pos_indices, dep_indices)

        # Flatten each of the embeddings into a single dimension
        batch_size = w_embeddings.shape[0]
        w_embeddings = w_embeddings.view(batch_size, -1)
        p_embeddings = p_embeddings.view(batch_size, -1)
        d_embeddings = d_embeddings.view(batch_size, -1)

        # Compute the output
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        output = self.sequence(torch.cat([w_embeddings, p_embeddings, d_embeddings, torch.ones(batch_size, 1).to(device)], dim=1).to(device))

        return output