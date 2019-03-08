from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.util import get_log_representation
from pm4py.objects.log.util import xes
from pm4py.util import constants
from pm4py.statistics.traces.log import case_statistics
from sklearn.linear_model import ElasticNet
from pm4py.objects.log.util.get_prefixes import get_log_with_log_prefixes

def train(log, parameters=None):
    if parameters is None:
        parameters = {}

    activity_key = parameters[
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY] if constants.PARAMETER_CONSTANT_ACTIVITY_KEY in parameters else xes.DEFAULT_NAME_KEY

    str_tr_attr, str_ev_attr, num_tr_attr, num_ev_attr = attributes_filter.select_attributes_from_log_for_tree(log)
    if activity_key not in str_ev_attr:
        str_ev_attr.append(activity_key)
    str_evsucc_attr = [activity_key]

    ext_log = get_log_with_log_prefixes(log)
    data, feature_names = get_log_representation.get_representation(ext_log, str_tr_attr, str_ev_attr, num_tr_attr,
                                                                    num_ev_attr, str_evsucc_attr=str_evsucc_attr)
    case_durations = case_statistics.get_all_casedurations(ext_log, parameters=parameters)

    print(len(data))
    print(len(case_durations))

    model = ElasticNet()
    model.fit(data, case_durations)

    print(case_durations)

    return {"str_tr_attr": str_tr_attr, "str_ev_attr": str_ev_attr, "num_tr_attr": num_tr_attr,
            "num_ev_attr": num_ev_attr, "str_evsucc_attr": str_evsucc_attr, "feature_names": feature_names}


def test(model, trace, parameters=None):
    if parameters is None:
        parameters = {}

    str_tr_attr = model["str_tr_attr"]
    str_ev_attr = model["str_ev_attr"]
    num_tr_attr = model["num_tr_attr"]
    num_ev_attr = model["num_ev_attr"]
    feature_names = model["feature_names"]

    return None
