import numpy as np
import torch
from feature_extraction import (NUM_DEPS, SHIFT, convert_string_to_sentence,
                                pos_prefix, punc_pos)
from general_utils import to_nltk_tree


def test_on_validation_data(model, dataset):
    print("Evaluating on dev set")
    compute_dependencies(model, dataset.valid_data, dataset)
    valid_UAS = get_UAS(dataset.valid_data)
    print("- dev UAS: {:.2f}".format(valid_UAS * 100.0))
    return valid_UAS


def parse_sentence(test_string, model, device, dataset, verbose=True):
    '''
    Computes the model's predictions for the stated dataset and returns both
    normal accuracies as well as UAS
    '''

    test_sentence = convert_string_to_sentence(test_string)

    preds = []
    answers = []

    sentences = [test_sentence]
    #rem_sentences = [sentence for sentence in sentences]
    [sentence.clear_prediction_dependencies() for sentence in sentences]
    [sentence.clear_children_info() for sentence in sentences]

    for batch_start in range(0, len(sentences),
                             dataset.model_config.batch_size):

        # The sentences we'll process
        batch_sentences = sentences[
            batch_start:batch_start + dataset.model_config.batch_size]

        # Which sentences still have some parsing steps to complete
        enable_features = []
        for sentence in batch_sentences:
            sentence.clear_prediction_dependencies()
            sentence.clear_children_info()
            # 0 -> the sentence has nothing left to parse
            enable_features.append(0 if len(sentence.stack) == 1 and
                                   len(sentence.buff) == 0 else 1)

        enable_count = np.count_nonzero(enable_features)

        # While there is still at least one sentence with parsing to do...
        while enable_count > 0:

            # get feature for each sentence
            # call predictions -> argmax
            # store dependency and left/right child
            # update state
            # repeat

            curr_sentences = []

            word_inputs_batch = []
            pos_inputs_batch = []
            dep_inputs_batch = []

            for i, sentence in enumerate(batch_sentences):
                # If we still have parsing to do for this sentence
                if enable_features[i] == 1:
                    curr_sentences.append(sentence)
                    inputs = dataset.feature_extractor.extract_for_current_state(
                        sentence, dataset.word2idx, dataset.pos2idx,
                        dataset.dep2idx)

                    word_inputs_batch.append(inputs[0])
                    pos_inputs_batch.append(inputs[1])
                    dep_inputs_batch.append(inputs[2])

            word_inputs_batch = torch.tensor(word_inputs_batch).to(device)
            pos_inputs_batch = torch.tensor(pos_inputs_batch).to(device)
            dep_inputs_batch = torch.tensor(dep_inputs_batch).to(device)

            # These are the raw outputs, which represent the activations for
            # prediction over valid transitions
            predictions = model(word_inputs_batch, pos_inputs_batch,
                                dep_inputs_batch)

            # print('predictions: ', predictions.size())

            legal_labels = np.asarray(
                [sentence.get_legal_labels() for sentence in curr_sentences],
                dtype=np.float32)
            legal_transitions = np.argmax(
                predictions.cpu().detach().numpy() + 1000 * legal_labels,
                axis=1)

            preds.extend(predictions.argmax(1).data.tolist())
            answers.extend([x.argmax() for x in legal_labels])

            # update the dep of the child token
            for (sentence, transition) in zip(curr_sentences,
                                              legal_transitions):
                if transition != SHIFT:
                    arc_label = transition - 1
                    if transition > NUM_DEPS:
                        arc_label -= NUM_DEPS
                    if transition <= NUM_DEPS:
                        dependent = sentence.stack[-2]
                    else:
                        dependent = sentence.stack[-1]
                    dependent.dep = dataset.idx2dep[arc_label]
            sentence, transition = curr_sentences[0], legal_transitions[0]
            if verbose:
                print("----")
                print("buffer: %s" % str([t.word for t in sentence.buff]))
                print("stack:  %s" % str([t.word for t in sentence.stack]))
                if transition == SHIFT:
                    print("action: shift")
                elif transition <= NUM_DEPS:
                    print("action: left arc, %s" %
                          dataset.idx2dep[transition - 1])
                else:
                    print("action: right arc, %s" %
                          dataset.idx2dep[transition - 1 - NUM_DEPS])

            # update left/right children so can be used for next feature vector
            [
                sentence.update_child_dependencies(transition)
                for (sentence,
                     transition) in zip(curr_sentences, legal_transitions)
                if transition != SHIFT
            ]

            # update state
            [
                sentence.update_state_by_transition(
                    legal_transition, gold=False)
                for (
                    sentence,
                    legal_transition) in zip(curr_sentences, legal_transitions)
            ]

            enable_features = [
                0
                if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1
                for sentence in batch_sentences
            ]
            enable_count = np.count_nonzero(enable_features)

        for sentence in batch_sentences:
            to_nltk_tree(sentence.Root, sentence.tokens).pretty_print()


def compute_dependencies(model, device, data, dataset):
    '''
    Computes the model's predictions for the stated dataset and returns both
    normal accuracies as well as UAS
    '''

    preds = []
    answers = []

    sentences = data
    #rem_sentences = [sentence for sentence in sentences]
    [sentence.clear_prediction_dependencies() for sentence in sentences]
    [sentence.clear_children_info() for sentence in sentences]

    for batch_start in range(0, len(sentences),
                             dataset.model_config.batch_size):

        # The sentences we'll process
        batch_sentences = sentences[
            batch_start:batch_start + dataset.model_config.batch_size]

        # Which sentences still have some parsing steps to complete
        enable_features = []
        for sentence in batch_sentences:
            sentence.clear_prediction_dependencies()
            sentence.clear_children_info()
            # 0 -> the sentence has nothing left to parse
            enable_features.append(0 if len(sentence.stack) == 1 and
                                   len(sentence.buff) == 0 else 1)

        enable_count = np.count_nonzero(enable_features)

        # While there is still at least one sentence with parsing to do...
        while enable_count > 0:

            # get feature for each sentence
            # call predictions -> argmax
            # store dependency and left/right child
            # update state
            # repeat

            curr_sentences = []

            word_inputs_batch = []
            pos_inputs_batch = []
            dep_inputs_batch = []

            for i, sentence in enumerate(batch_sentences):
                # If we still have parsing to do for this sentence
                if enable_features[i] == 1:
                    curr_sentences.append(sentence)
                    inputs = dataset.feature_extractor.extract_for_current_state(
                        sentence, dataset.word2idx, dataset.pos2idx,
                        dataset.dep2idx)

                    word_inputs_batch.append(inputs[0])
                    pos_inputs_batch.append(inputs[1])
                    dep_inputs_batch.append(inputs[2])

            word_inputs_batch = torch.tensor(word_inputs_batch).to(device)
            pos_inputs_batch = torch.tensor(pos_inputs_batch).to(device)
            dep_inputs_batch = torch.tensor(dep_inputs_batch).to(device)

            # These are the raw outputs, which represent the activations for
            # prediction over valid transitions
            predictions = model(word_inputs_batch, pos_inputs_batch,
                                dep_inputs_batch)

            # print('predictions: ', predictions.size())

            legal_labels = np.asarray(
                [sentence.get_legal_labels() for sentence in curr_sentences],
                dtype=np.float32)
            legal_transitions = np.argmax(
                predictions.cpu().detach().numpy() + 1000 * legal_labels,
                axis=1)

            preds.extend(predictions.argmax(1).data.tolist())
            answers.extend([x.argmax() for x in legal_labels])

            # update left/right children so can be used for next feature vector
            [
                sentence.update_child_dependencies(transition)
                for (sentence,
                     transition) in zip(curr_sentences, legal_transitions)
                if transition != SHIFT
            ]

            # update state
            [
                sentence.update_state_by_transition(
                    legal_transition, gold=False)
                for (
                    sentence,
                    legal_transition) in zip(curr_sentences, legal_transitions)
            ]

            enable_features = [
                0
                if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1
                for sentence in batch_sentences
            ]
            enable_count = np.count_nonzero(enable_features)

        # Once, we've finished all the steps from parsing, reset the stack and
        # buffer for the next time we might see this sentence.
        [sentence.reset_to_initial_state() for sentence in batch_sentences]

    # print accuracy at end

    # correct = np.array(preds, dtype=int) == np.array(answers, dtype=int)
    # print('Validation acc: %1.3f' % (correct.sum() / len(correct)))


def get_UAS(data):
    correct_tokens = 0
    all_tokens = 0
    punc_token_pos = [pos_prefix + each for each in punc_pos]
    for sentence in data:
        # reset each predicted head before evaluation
        [token.reset_predicted_head_id() for token in sentence.tokens]

        head = [-2] * len(sentence.tokens)
        # assert len(sentence.dependencies) == len(sentence.predicted_dependencies)
        for h, t, in sentence.predicted_dependencies:
            head[t.token_id] = h.token_id

        non_punc_tokens = [
            token for token in sentence.tokens
            if token.pos not in punc_token_pos
        ]
        correct_tokens += sum([
            1 if token.head_id == head[token.token_id] else 0
            for (_, token) in enumerate(non_punc_tokens)
        ])

        # all_tokens += len(sentence.tokens)
        all_tokens += len(non_punc_tokens)

    UAS = correct_tokens / float(all_tokens)
    return UAS
