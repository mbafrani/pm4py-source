from copy import deepcopy

import numpy as np
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.util import get_log_representation
from pm4py.objects.log.util import sorting
from pm4py.objects.log.util import xes
from pm4py.objects.log.util.get_prefixes import get_log_with_log_prefixes
from pm4py.statistics.traces.log import case_statistics
from pm4py.util import constants


def get_trace_rep_rnn(trace, dictionary_features, str_ev_attr, str_evsucc_attr, max_len_trace):
    X = []
    for index, event in enumerate(trace):
        added = False
        ev_vector = [0] * len(dictionary_features)
        for attribute_name in event:
            attribute_value = event[attribute_name]
            rep = "event:" + str(attribute_name) + "@" + str(attribute_value)
            if rep in dictionary_features:
                ev_vector[dictionary_features[rep]] = 1
        if index < len(trace) - 1:
            next_event = trace[index + 1]
            for attribute_name in event:
                if attribute_name in next_event:
                    attribute_value_1 = event[attribute_name]
                    attribute_value_2 = next_event[attribute_name]
                    rep = "succession:" + str(attribute_name) + "@" + str(attribute_value_1) + "#" + str(
                        attribute_value_2)
                    if rep in dictionary_features:
                        ev_vector[dictionary_features[rep]] = 1
        X.append(ev_vector)
    j = len(trace)
    while j < max_len_trace:
        X.append(X[-1])
        j = j + 1
    X = np.transpose(np.asmatrix(X))
    X = X.tolist()

    return X


def get_log_rep_rnn(log, dictionary_features, str_ev_attr, str_evsucc_attr, max_len_trace):
    X_train = []
    for trace in log:
        rep = get_trace_rep_rnn(trace, dictionary_features, str_ev_attr, str_evsucc_attr, max_len_trace)
        if rep:
            X_train.append(rep)

    return X_train


def group_remaining_time(change_indexes, remaining_time, max_len_trace):
    rem_time_grouped = []
    j = 0
    for ct in change_indexes:
        rem = []
        added = False

        for i in range(len(ct)):
            rem.append(remaining_time[j])
            if i == max_len_trace - 1:
                rem_time_grouped.append(deepcopy(rem))
                added = True
            elif i == len(ct) - 1 and not added:
                while len(rem) < max_len_trace:
                    rem.append(rem[-1])
                rem_time_grouped.append(deepcopy(rem))
            j = j + 1
    return rem_time_grouped


def train(log, parameters=None):
    if parameters is None:
        parameters = {}

    parameters["enable_sort"] = False
    activity_key = parameters[
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY] if constants.PARAMETER_CONSTANT_ACTIVITY_KEY in parameters else xes.DEFAULT_NAME_KEY
    timestamp_key = parameters[
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY] if constants.PARAMETER_CONSTANT_TIMESTAMP_KEY in parameters else xes.DEFAULT_TIMESTAMP_KEY

    log = sorting.sort_timestamp(log, timestamp_key)
    max_len_trace = max([len(trace) for trace in log])

    str_tr_attr, str_ev_attr, num_tr_attr, num_ev_attr = attributes_filter.select_attributes_from_log_for_tree(log)
    if activity_key not in str_ev_attr:
        str_ev_attr.append(activity_key)
    str_evsucc_attr = [activity_key]

    dictionary_features = {}
    data, feature_names = get_log_representation.get_representation(log, str_tr_attr, str_ev_attr, num_tr_attr,
                                                                    num_ev_attr, str_evsucc_attr=str_evsucc_attr)
    for index, value in enumerate(feature_names):
        dictionary_features[value] = index

    ext_log, change_indexes = get_log_with_log_prefixes(log)
    print("max_len_trace = ", max_len_trace)

    case_durations = case_statistics.get_all_casedurations(ext_log, parameters=parameters)

    X_train = get_log_rep_rnn(log, dictionary_features, str_ev_attr, str_evsucc_attr, max_len_trace)
    change_indexes_flattened = [y for x in change_indexes for y in x]
    remaining_time = [-case_durations[i] + case_durations[change_indexes_flattened[i]] for i in
                      range(len(case_durations))]

    y_train = np.array(group_remaining_time(change_indexes, remaining_time, max_len_trace))
    # X_train = np.asarray(X_train).reshape((y_train.shape[0], len(feature_names), len(X_train)/()))
    X_train = np.array(X_train)
    print(X_train)
    print(y_train)

    print(X_train.shape)
    print(y_train.shape)
    print(len(ext_log))
    input()

    in_out_neurons =  X_train.shape[2]
    hidden_neurons = in_out_neurons * 10
    input_shape = (X_train.shape[1], X_train.shape[2])

    print("in_out_neurons = ",in_out_neurons)

    print(input_shape)

    model = Sequential()
    model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=input_shape))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    model.fit(X_train, y_train, batch_size=700, nb_epoch=100, validation_split=0.05)


def test(model, obj, parameters=None):
    if parameters is None:
        parameters = {}
