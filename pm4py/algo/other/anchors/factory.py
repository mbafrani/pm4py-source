from pm4py.algo.other.anchors.versions import classic

CLASSIC = "classic"

VERSIONS = {CLASSIC: classic.ClassicAnchorClassification}


def apply(log, target, classes, variant=CLASSIC, parameters=None):
    if parameters is None:
        parameters = {}

    return VERSIONS[variant](log, target, classes, parameters=parameters)
