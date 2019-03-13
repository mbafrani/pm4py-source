import math
from copy import deepcopy

import numpy as np
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.log import EventLog
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


def normalize_remaining_time(rem_time_grouped):
    ret = []
    max_value = -10000000
    for lst in rem_time_grouped:
        max_lst = max(lst)
        max_value = max(max_value, max_lst)
    log_max_value = math.log(1.0 + max_value)
    for lst in rem_time_grouped:
        ret.append([])
        for val in lst:
            ret[-1].append(-1.0 + 2.0 * (math.log(val + 1.0) / log_max_value))
    return ret, log_max_value


def reconstruct_value(y, log_max_value):
    if y < -1:
        y = -1
    return math.exp((y + 1.0) / 2.0 * log_max_value) - 1


def get_X_from_log(log, feature_names, str_ev_attr, str_evsucc_attr, max_len_trace):
    dictionary_features = {}
    for index, value in enumerate(feature_names):
        dictionary_features[value] = index
    X = get_log_rep_rnn(log, dictionary_features, str_ev_attr, str_evsucc_attr, max_len_trace)
    X = np.array(X)

    return X


def train(log, parameters=None):
    if parameters is None:
        parameters = {}
    default_epochs = parameters["default_epochs"] if "default_epochs" in parameters else 15
    parameters["enable_sort"] = False
    activity_key = parameters[
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY] if constants.PARAMETER_CONSTANT_ACTIVITY_KEY in parameters else xes.DEFAULT_NAME_KEY
    timestamp_key = parameters[
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY] if constants.PARAMETER_CONSTANT_TIMESTAMP_KEY in parameters else xes.DEFAULT_TIMESTAMP_KEY
    log = sorting.sort_timestamp(log, timestamp_key)
    max_len_trace = max([len(trace) for trace in log])
    ext_log, change_indexes = get_log_with_log_prefixes(log)
    case_durations = case_statistics.get_all_casedurations(ext_log, parameters=parameters)
    change_indexes_flattened = [y for x in change_indexes for y in x]
    remaining_time = [-case_durations[i] + case_durations[change_indexes_flattened[i]] for i in
                      range(len(case_durations))]
    y_orig = group_remaining_time(change_indexes, remaining_time, max_len_trace)
    y, log_max_value = normalize_remaining_time(y_orig)
    y = np.array(y)
    str_tr_attr, str_ev_attr, num_tr_attr, num_ev_attr = attributes_filter.select_attributes_from_log_for_tree(log)
    if activity_key not in str_ev_attr:
        str_ev_attr.append(activity_key)
    str_evsucc_attr = [activity_key]
    data, feature_names = get_log_representation.get_representation(log, str_tr_attr, str_ev_attr, num_tr_attr,
                                                                    num_ev_attr, str_evsucc_attr=str_evsucc_attr)
    X = get_X_from_log(log, feature_names, str_ev_attr, str_evsucc_attr, max_len_trace)
    in_out_neurons = X.shape[2]
    hidden_neurons = int(in_out_neurons * 7.5)
    input_shape = (X.shape[1], X.shape[2])
    model = Sequential()
    model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=input_shape))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(X, y, batch_size=X.shape[1], nb_epoch=default_epochs, validation_split=0.2)
    return {"str_tr_attr": str_tr_attr, "str_ev_attr": str_ev_attr, "num_tr_attr": num_tr_attr,
            "num_ev_attr": num_ev_attr, "str_evsucc_attr": str_evsucc_attr, "feature_names": feature_names,
            "remaining_time": remaining_time, "regr": model, "max_len_trace": max_len_trace,
            "log_max_value": log_max_value}


def test(model, obj, parameters=None):
    if parameters is None:
        parameters = {}
    str_ev_attr = model["str_ev_attr"]
    str_evsucc_attr = model["str_evsucc_attr"]
    feature_names = model["feature_names"]
    regr = model["regr"]
    max_len_trace = model["max_len_trace"]
    log_max_value = model["log_max_value"]
    if type(obj) is EventLog:
        log = obj
    else:
        log = EventLog([obj])
    max_len_trace_test_log = max([len(trace) for trace in log])
    if max_len_trace_test_log > max_len_trace:
        raise Exception(
            "cannot predict when the maximum length of the test log is greater than the maximum length of the training log")
    X = get_X_from_log(log, feature_names, str_ev_attr, str_evsucc_attr, max_len_trace)
    y = regr.predict(X)
    if len(log) == 1:
        return reconstruct_value(y[0][len(log[0]) - 1], log_max_value)
