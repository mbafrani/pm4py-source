
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from collections import defaultdict
from pm4py.objects.log import transform
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.exporter.csv import factory as csv_exporter
from pyvis.network import Network

class ResourcePerformace:

    # Get event log (csv, xes format), Convert both to csv file
	def get_input_file(self):
        event_log_address = input("Event Log:")
        log_format = event_log_address.split('.')[-1]

        if str(log_format) == 'csv':
            event_log = pd.read_csv(event_log_address)

        elif str(log_format) == 'xes':
            xes_log = xes_importer.import_log(event_log_address)
            event_log = transform.transform_trace_log_to_event_log(xes_log)
            csv_exporter.export_log(event_log, "event_log.csv")
            event_log = pd.read_csv("event_log.csv")

        event_log_attributes = event_log.columns

        return event_log

    # Create DataFrame including 
	def create_matrix_resource(self, event_log):

        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'])
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        event_duration = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        event_log['Event Duration'] = event_duration
      
		# create adjancy matrix of activities
        resource_matrix = pd.DataFrame(
            np.zeros(shape=(event_log['Resource'].nunique(), event_log['Resource'].nunique())),
            columns=event_log['Resource'].unique(), index=event_log['Resource'].unique())
        temp_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby('Case ID')
        trace = {}

        for case, casegroup in temp_log:
            trace.update({case: casegroup['Resource'].values})

        for key, val in trace.items():
            i = 0
            while i < (len(val) - 1):
                resource_matrix[val[i + 1]][val[i]] += 1
                i += 1

        return resource_matrix
	
	# Create Matrix of Activities (Adjancy Matrix for Directly Follows Graph) 
	# In: event log, Out: Adjancy Matrix of Activities
    def create_matrix(self, event_log):
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'])
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        event_duration = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        event_log['Event Duration'] = event_duration
        
		# create activity duration dictionary		
		act_dur_dict = {}
        temp_act_log = event_log.groupby(['Activity'])
        for kact, vact in temp_act_log:
            act_dur_dict[kact] = vact['Event Duration'].mean()
			
		# create adjancy matrix of activities
        matrix = pd.DataFrame(np.zeros(shape=(event_log['Activity'].nunique(), event_log['Activity'].nunique())),
                              columns=event_log['Activity'].unique(), index=event_log['Activity'].unique())
        temp_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby('Case ID')
        trace = {}

        for case, casegroup in temp_log:
            trace.update({case: casegroup['Activity'].values})

        for key, val in trace.items():
            i = 0
            while i < (len(val) - 1):
                matrix[val[i + 1]][val[i]] += 1
                i += 1
        return matrix,act_dur_dict

	# Create Two Dataframe including Resource duration for each activity, Frequency of each resource doing each activity 
	# In: event log, Out: Two Dataframes of Resource activity relation (Frequency and Duration) 
    def find_resource(self, event_log):

        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'])
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'])
        event_duration = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        event_log['Event Duration'] = event_duration

        freq_act_res_matrix = pd.DataFrame(
            np.zeros(shape=(len(event_log['Resource'].unique()), len(event_log['Activity'].unique()))),
            columns=event_log['Activity'].unique(), index=event_log['Resource'].unique())
        dur_act_res_matrix = pd.DataFrame(
            np.zeros(shape=(len(event_log['Resource'].unique()), len(event_log['Activity'].unique()))),
            columns=event_log['Activity'].unique(), index=event_log['Resource'].unique())

        act_groupy = event_log.groupby('Activity')
        for name, group in act_groupy:
            resgroup = group.groupby('Resource')['Event Duration']
            res_per_act_freq = resgroup.size()
			res_per_act_freq = round(res_per_act_freq,2)
            res_per_act_sum = resgroup.sum()

            for res in res_per_act_freq.keys():
                freq_act_res_matrix[name][res] = res_per_act_freq.get(res)
                if res_per_act_freq.get(res) != 0 and res_per_act_sum.get(res) != 0:
                    dur_act_res_matrix[name][res] = pd.to_timedelta(
                        (res_per_act_sum.get(res)) / res_per_act_freq.get(res)).seconds // 3600

        return freq_act_res_matrix, dur_act_res_matrix

    # Create Matrix of Activities (Adjancy Matrix for Directly Follows Graph) 
	# In: Activity Adjancy Matrix, Resource Activity Duration dataframe, Resource Activity Frequency dataframe 
	#Out: Graph of performance of each resource for each activity on the Directly Follows Graph
	def draw_matrix(self, matrix, freq_act_res_matrix, dur_act_res_matrix,act_dur_dict):
        
        # Build a dataframe with 4 connections
        matrix[matrix<0] = 0
        G = Network()
        act_dur_sum = np.mean(pd.to_timedelta(act_dur_dict.values()).total_seconds()/3600)
        matrixt = matrix.T

        for act in matrix.columns:
            act_dur_per = pd.to_timedelta(act_dur_dict[act]).total_seconds()/3600
            if act_dur_per > (act_dur_sum + 1/2*act_dur_sum):
                act_color = 'darkorchid'
            if act_dur_per < (act_dur_sum - 1/2*act_dur_sum):
                act_color='plum'
            else:
                act_color='mediumpurple'
            G.add_node(act, shape='box',label=str(act)+ '\n' +str(act_dur_dict[act]),color=act_color)
            temp_in = matrix[act]
            sum_temp_in_values = np.sum(temp_in.values)
            for intemp in temp_in.iteritems():
                nact_dur_per = pd.to_timedelta(act_dur_dict[intemp[0]]).total_seconds() / 3600
                if nact_dur_per > 2 * (act_dur_sum / 3):
                    act_color = 'darkorchid'
                if nact_dur_per < act_dur_sum / 3:
                    act_color = 'plum'
                else:
                    act_color = 'mediumpurple'
                if intemp[1]/sum_temp_in_values> 0.1:
                    #tem_max_in =temp_in.idxmax()
                    G.add_node(intemp[0], shape='box',color = act_color,label=str(intemp[0])+ '\n' +str(act_dur_dict[intemp[0]]))
                    G.add_edge(act,intemp[0])
           
        i = 0

        act_act_res_dict = defaultdict(dict)
        for ac in freq_act_res_matrix.columns:
            act_res_dict = defaultdict(list)
            temp_freq_res = freq_act_res_matrix[ac]
            temp_dur_res = dur_act_res_matrix[ac]
            temp_dur_res = temp_dur_res[temp_dur_res != 0]
        
            res_dur_mean = np.mean(temp_dur_res)
            res_fer_var = np.std(temp_freq_res)
            res_fer_mean = np.mean(temp_freq_res)

            for res in temp_dur_res.keys():
                if temp_dur_res.get(res) >= res_dur_mean + res_dur_mean / 2:
                    act_res_dict[res].append("bad" + "/" + str(temp_dur_res.get(res)))
                elif res_dur_mean - res_dur_mean / 2 < temp_dur_res.get(res) < res_dur_mean + res_dur_mean / 2:
                    act_res_dict[res].append("average" + "/" + str(temp_dur_res.get(res)))
                elif abs(res_dur_mean - res_dur_mean / 2) >= temp_dur_res.get(res):
                    act_res_dict[res].append("good" + "/" + str(temp_dur_res.get(res)))

                sum_temp = np.sum(temp_freq_res)

                act_res_dict[res].append(100 * (temp_freq_res.get(res) / np.sum(temp_freq_res)))
                act_act_res_dict[ac].update(act_res_dict)

        for a, av in act_act_res_dict.items():
            resources = av.keys()
            for resource in resources:
                if len(av.get(resource)) > 1 and av.get(resource)[1] >= 1:
                    resource_duration = av.get(resource)[0]
                    resource_size = av.get(resource)[1]

                    if resource_duration.split("/")[0] == "good":
                        r_color = "#00ff1e"
                    if resource_duration.split("/")[0] == "average":
                        r_color = "#cdc9c9"
                    if resource_duration.split("/")[0] == "bad":
                        r_color = "#ff0000"

                    r_size = resource_size


                    if r_color == "#cdc9c9":
                        rb_color = "darkgrey"
                    elif r_color == "#ff0000":
                        rb_color = "pink"
                    elif r_color == "#00ff1e":
                        rb_color = "palegreen"

                    highlight = {'border': r_color, 'background': r_color}
                    #                    if resource_size > 1:

                    G.add_node(str(resource) + str(i), labelHighlightBold=True,
                               title=str(round(resource_size, 2)) + "and" + str(resource_duration)+"h",
                               label=resource,
                               color={'border': rb_color, 'background': rb_color, 'highlight': highlight}, size=r_size)
                    G.add_edge(str(resource) + str(i), a)

            i += 1

        G.show_buttons(filter_=['physics', 'nodes', 'edges'])

        G.show("mygraph.html")
        plt.show()
        return


if __name__ == "__main__":
    resper = ResourcePerformace()
    event_log = resper.get_input_file()
    matrix,act_dur_dict = resper.create_matrix(event_log)
    freq_act_res_matrix, dur_act_res_matrix = resper.find_resource(event_log)
    resper.draw_matrix(matrix, freq_act_res_matrix, dur_act_res_matrix,act_dur_dict)