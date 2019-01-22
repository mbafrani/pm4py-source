def get_gateway_map(bpmn_graph):
    gateway_map = {}
    edges_map = {}

    dg = bpmn_graph.diagram_graph
    for e in dg.edges:
        edge = dg.edges[e]
        edges_map[edge['id']] = edge
    for n in dg.nodes:
        node = dg.nodes[n]
        node_type = node["type"]
        if node_type == "exclusiveGateway":
            node_incoming = []
            node_outgoing = []
            for x in node["incoming"]:
                if x in edges_map and edges_map[x]["sourceRef"] in dg.nodes:
                    node_incoming.append(dg.nodes[edges_map[x]["sourceRef"]])
            for x in node["outgoing"]:
                if x in edges_map and edges_map[x]["targetRef"] in dg.nodes:
                    node_outgoing.append(dg.nodes[edges_map[x]["targetRef"]])
            incoming_tasks = [x for x in node_incoming if "task" in x["type"].lower()]
            if len(node_incoming) == 1 and len(incoming_tasks) == 1 and len(node_outgoing) > 1:
                task_nodes = [x for x in node_outgoing if "task" in x["type"].lower()]
                gateway_nodes = [x for x in node_outgoing if "gateway" in x["type"].lower()]
                other_nodes = [x for x in node_outgoing if x not in task_nodes and x not in gateway_nodes]
                if len(other_nodes) == 0 and task_nodes:
                    if gateway_nodes and len(gateway_nodes) == 1 and len(task_nodes) == 1:
                        gateway_map[n] = {"type": "gateway", "source": incoming_tasks[0]["node_name"], "edges": {}}
                        for task in task_nodes:
                            gateway_map[n]["edges"][task["node_name"]] = task["incoming"][0]
                    elif len(gateway_nodes) == 0 and len(task_nodes) > 1:
                        gateway_map[n] = {"type": "onlytasks", "source": incoming_tasks[0]["node_name"], "edges": {}}
                        for task in task_nodes:
                            gateway_map[n]["edges"][task["node_name"]] = task["incoming"][0]

    for n in gateway_map:
        for t in gateway_map[n]["edges"]:
            print(t, gateway_map[n]["source"], gateway_map[n]["edges"][t])

    return gateway_map