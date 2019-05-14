from pm4py.algo.other.anchors.versions import classic

CLASSIC = "classic"

VERSIONS = {CLASSIC: classic.ClassicAnchorClassification}


def apply(log, target, classes, variant=CLASSIC, parameters=None):
    """
    Initialize the Anchors classification

    Parameters
    ------------
    log
        Log
    target
        Numerical target of the classification
    classes
        Classes (of the classification)
    parameters
        Parameters of the algorithm
    variant
        Variant of the algorithm to use, possible values: classic

    Returns
    ------------
    anchors
        Anchors classifier
    """
    if parameters is None:
        parameters = {}

    return VERSIONS[variant](log, target, classes, parameters=parameters)
