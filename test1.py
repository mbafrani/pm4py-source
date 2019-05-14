import os

from pm4py.algo.other.decisiontree.applications import root_cause_part_duration
from pm4py.algo.other.decisiontree.applications.root_cause_part_duration import get_data_classes_root_cause_analysis
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.visualization.decisiontree import factory as dt_vis_factory
from pm4py.algo.other.anchors import factory as anchors_factory
from pm4py.objects.log.util import get_log_representation, get_prefixes


def execute_script():
    log = xes_importer.apply(os.path.join("tests", "input_data", "roadtraffic100traces.xes"))

    transf_log, traces_interlapsed_time_to_act = get_prefixes.get_log_traces_until_activity(log, "Payment")

    data, feature_names, target, classes = get_data_classes_root_cause_analysis(log, "Payment")

    anchors = anchors_factory.apply(transf_log, target, classes)
    anchors.train()

    for trace in transf_log:
        pred = anchors.predict(trace)
        print(pred)

        explain = anchors.explain(trace)
        print(explain)

        input()

    #gviz = dt_vis_factory.apply(clf, feature_names, classes, parameters={"format": "svg"})
    #dt_vis_factory.view(gviz)


if __name__ == "__main__":
    execute_script()