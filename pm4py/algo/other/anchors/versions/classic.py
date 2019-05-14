from pm4py.algo.other.anchors.interface.anchor_classification import AnchorClassification
from sklearn.linear_model import PassiveAggressiveClassifier
from anchor import anchor_tabular
from pm4py.objects.log.util import get_log_representation
from pm4py.objects.log.log import EventLog


class ClassicAnchorClassification(AnchorClassification):
    def __init__(self, log, target, classes, parameters=None):
        if parameters is None:
            parameters = {}

        self.parameters = parameters

        self.log = log
        self.data, self.feature_names, self.str_tr_attr, self.str_ev_attr, self.num_tr_attr, self.num_ev_attr = get_log_representation.get_default_representation_with_attribute_names(
            self.log)
        self.target = target
        self.classes = classes

        AnchorClassification.__init__(self, self.log, target, classes, parameters=None)

    def train(self):
        self.classifier = PassiveAggressiveClassifier()
        self.explainer = anchor_tabular.AnchorTabularExplainer(self.classes, self.feature_names, self.data,
                                                               categorical_names={})
        self.explainer.fit(self.data, self.feature_names, self.data, self.feature_names)
        self.classifier.fit(self.explainer.encoder.transform(self.data), self.target)

    def predict(self, trace):
        log = EventLog()
        log.append(trace)
        data, feature_names = get_log_representation.get_representation(log, self.str_tr_attr, self.str_ev_attr,
                                                                        self.num_tr_attr, self.num_ev_attr,
                                                                        feature_names=self.feature_names)

        prediction = self.classifier.predict(data)[0]
        return prediction

    def explain(self, trace, threshold=-0.01):
        log = EventLog()
        log.append(trace)
        data, feature_names = get_log_representation.get_representation(log, self.str_tr_attr, self.str_ev_attr,
                                                                        self.num_tr_attr, self.num_ev_attr,
                                                                        feature_names=self.feature_names)

        print(data[0])

        exp = self.explainer.explain_instance(data[0], self.classifier.predict, threshold=0.95)

        return ' AND '.join(exp.names())
