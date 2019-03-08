from pm4py.algo.prediction.versions import elasticnet

ELASTICNET = "elasticnet"

VERSIONS_TRAIN = {ELASTICNET: elasticnet.train}
VERSIONS_TEST = {ELASTICNET: elasticnet.test}


def train(log, variant=ELASTICNET, parameters=None):
    return VERSIONS_TRAIN[variant](log, parameters=parameters)


def test(model, trace, variant=ELASTICNET, parameters=None):
    return VERSIONS_TEST[variant](model, trace, parameters=parameters)
