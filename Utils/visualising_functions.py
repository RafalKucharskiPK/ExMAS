import itertools
import sys
import warnings
import pathlib

import seaborn
import seaborn as sns
import Utils.utils_topology as utils_topology
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import datetime
import json
from dotmap import DotMap
from netwulf import visualize
import netwulf as nw
import matplotlib.ticker as mtick
import os
import tkinter as tk
import multiprocessing as mp
import matplotlib as mpl
from matplotlib.lines import Line2D

import scienceplots

plt.style.use(['science', 'no-latex'])
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_parameters(path, time_correction=False):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    if time_correction:
        config['t0'] = pd.Timestamp('15:00')

    return config


def create_figs_folder(config):
    try:
        os.mkdir(config.path_results + "figs")
    except OSError as error:
        print(error)
        print('overwriting current files in the folder')
    config.path_results += '/'


def config_initialisation(path, date, sblts_exmas="exmas"):
    topological_config = get_parameters(path)
    topological_config.path_results = 'data/results/' + date + '/'
    topological_config.date = date
    topological_config.sblts_exmas = sblts_exmas
    init_config = None
    try:
        init_config = get_parameters(topological_config.initial_parameters)
    except:
        pass
    if init_config is None:
        try:
            init_config = get_parameters(
                os.path.join(pathlib.Path(os.getcwd()).parent.absolute(), topological_config.initial_parameters))
        except:
            pass

    if init_config is not None:
        for key in init_config.keys():
            if key not in topological_config.keys():
                topological_config[key] = init_config[key]

    return topological_config


def load_data(config, other_var=None):
    if other_var is not None:
        config.path_results = other_var
    with open(config.path_results + '/rep_graphs_' + config.date + '.obj', 'rb') as file:
        rep_graphs = pickle.load(file)

    with open(config.path_results + '/dotmap_list_' + config.date + '.obj', 'rb') as file:
        dotmap_list = pickle.load(file)

    with open(config.path_results + '/all_graphs_list_' + config.date + '.obj', 'rb') as file:
        all_graphs_list = pickle.load(file)

    return rep_graphs, dotmap_list, all_graphs_list


def draw_bipartite_graph(graph, max_weight, config=None, save=False, saving_number=0, width_power=1,
                         figsize=(5, 12), dpi=100, node_size=1, batch_size=147, plot=True, date=None,
                         default_edge_size=1, name=None, colour_specific_node=None):
    # G1 = nx.convert_node_labels_to_integers(graph)
    G1 = graph
    x = G1.nodes._nodes
    l = []
    r = []
    for i in x:
        j = x[i]
        if j['bipartite'] == 1:
            l.append(i)
        else:
            r.append(i)

    if nx.is_weighted(G1):
        dict_weights = {tuple(edge_data[:-1]): edge_data[-1]["weight"] for edge_data in G1.edges(data=True)}
        r_weighted = {v[-1]: dict_weights[v] for v in dict_weights.keys() if v[-1] in r}
        r_weighted_sorted = {k: v for k, v in sorted(r_weighted.items(), key=lambda x: x[1])}
        r = list(r_weighted_sorted.keys())

    colour_list = len(l) * ['g'] + len(r) * ['b']

    pos = nx.bipartite_layout(G1, l)

    new_pos = dict()
    for num, key in enumerate(pos.keys()):
        if num <= batch_size - 1:
            new_pos[key] = pos[key]
        else:
            new_pos[r[num - batch_size]] = pos[key]

    if colour_specific_node is not None:
        assert isinstance(colour_specific_node, int), "Passed node number is not an integer"
        colour_list[colour_specific_node] = "r"

    plt.figure(figsize=figsize, dpi=dpi)

    nx.draw_networkx_nodes(G1, pos=new_pos, node_color=colour_list, node_size=node_size)

    if nx.is_weighted(G1):
        for weight in range(1, max_weight + 1):
            edge_list = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] == weight]
            nx.draw_networkx_edges(G1, new_pos, edgelist=edge_list,
                                   width=default_edge_size * np.power(weight, width_power)
                                         / np.power(max_weight, width_power))
    else:
        if colour_specific_node is None:
            nx.draw_networkx_edges(G1, new_pos, edgelist=G1.edges, width=default_edge_size)
        else:
            assert isinstance(colour_specific_node, int), "Passed node number is not an integer"
            colour_list = []
            for item in G1.edges:
                if item[0] == colour_specific_node or item[1] == colour_specific_node:
                    colour_list.append("red")
                else:
                    colour_list.append("black")
            nx.draw_networkx_edges(G1, new_pos, edgelist=G1.edges, width=default_edge_size / 5, edge_color=colour_list)

    if save:
        if date is None:
            date = str(datetime.date.today().strftime("%d-%m-%y"))
        else:
            date = str(date)
        if name is None:
            plt.savefig(config.path_results + "temp/graph_" + date + "_no_" + str(saving_number) + ".png",
                        transparent=True, pad_inches=0)
        else:
            plt.savefig(config.path_results + "temp/" + name + ".png")
    if plot:
        plt.show()


def graph_visualisation_with_netwulf(all_graphs=None, rep_graphs=None, graph_list=None, show_dialogue=True):
    if graph_list is None:
        graph_list = ["single_pairs_shareability", "single_pairs_matching",
                      "full_pairs_shareability", "full_pairs_matching"]

    if all_graphs is None:
        for g in ["single_pairs_shareability", "single_pairs_matching"]:
            if g in graph_list:
                graph_list.remove(g)
    else:
        no_nodes = len(all_graphs[0]["pairs_matching"].nodes)

    if rep_graphs is None:
        for g in ["full_pairs_shareability", "full_pairs_matching"]:
            if g in graph_list:
                graph_list.remove(g)
    else:
        no_nodes = len(rep_graphs["pairs_matching"].nodes)

    try:
        no_nodes
    except NameError:
        warnings.warn("Error trying to read number of nodes")
    else:
        pass

    for g in graph_list:
        if show_dialogue:
            text = "The following network is " + g.upper() + \
                   ". \n Please click 'Post to python' in the browser when investigated." + \
                   "\n Default name for the graph is: " + g + "_" + str(no_nodes)
            window = tk.Tk()
            lbl = tk.Label(window, text="Input")
            lbl.pack()
            txt = tk.Text(window, width=100, height=20)
            txt.pack()
            txt.insert("1.0", text)
            button = tk.Button(window, text="Show", command=window.destroy)
            button.pack()
            window.mainloop()

        if g == "single_pairs_shareability":
            graph = all_graphs[0]['pairs_shareability']
        elif g == "single_pairs_matching":
            graph = all_graphs[0]['pairs_matching']
        elif g == "full_pairs_shareability":
            graph = rep_graphs['pairs_shareability']
        elif g == "full_pairs_matching":
            graph = rep_graphs['pairs_matching']
        else:
            raise Warning("incorrect graph_list")

        visualize(graph, config=json.load(open('../Topology/data/configs/netwulf_config.json')))


def visualise_graph_evolution(dotmap_list, topological_config, num_list=None, node_size=1, dpi=80,
                              fig_size=(10, 24), plot=False, width_power=1, save=True):
    if num_list is None:
        list([1, 5, 10, 100, 900])

    for num in num_list:
        if num == 1:
            obj = [dotmap_list[1]]
        else:
            obj = dotmap_list[:num]
        draw_bipartite_graph(utils_topology.analyse_edge_count(obj, topological_config,
                                                               list_types_of_graph=['bipartite_matching'],
                                                               logger_level='WARNING')[
                                 'bipartite_matching'],
                             num, node_size=node_size, dpi=dpi, figsize=fig_size, plot=plot, width_power=width_power,
                             config=topological_config, save=save, saving_number=num, date=topological_config.date)


def kpis_gain(dotmap_list, topological_config, max_ticks=5, bins=20, y_max=20):
    sblts_exmas = topological_config.sblts_exmas
    str_for_end = "_" + str(len(dotmap_list[0][sblts_exmas].requests))
    multiplier = 100
    res = []

    data = [multiplier * (x[sblts_exmas].res.PassUtility_ns - x[sblts_exmas].res.PassUtility) / x[
        sblts_exmas].res.PassUtility_ns for x in dotmap_list]
    res.append(["relative_pass_utility", np.mean(data), np.std(data),
                np.nanpercentile(data, 5), np.nanpercentile(data, 95)])
    ax = sns.histplot(data, bins=bins)
    ax.axvline(4.5, ls=":")
    ax.set(xlabel=None, ylabel=None, yticklabels=[])
    plt.ylim(0, y_max)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.locator_params(axis='x', nbins=max_ticks)
    plt.savefig(topological_config.path_results + "figs/relative_pass_utility" + str_for_end + ".png")
    plt.close()

    data = [multiplier * (x[sblts_exmas].res.PassHourTrav - x[sblts_exmas].res.PassHourTrav_ns) / x[
        sblts_exmas].res.PassHourTrav_ns for x in dotmap_list]
    res.append(["relative_pass_hours", np.mean(data), np.std(data),
                np.nanpercentile(data, 5), np.nanpercentile(data, 95)])
    ax = sns.histplot(data, bins=bins)
    ax.axvline(9.8, ls=":")
    ax.set(xlabel=None, ylabel=None, yticklabels=[])
    plt.ylim(0, y_max)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.locator_params(axis='x', nbins=max_ticks)
    plt.savefig(topological_config.path_results + "figs/relative_pass_hours" + str_for_end + ".png")
    plt.close()

    data = [multiplier * (x[sblts_exmas].res.VehHourTrav_ns - x[sblts_exmas].res.VehHourTrav) / x[
        sblts_exmas].res.VehHourTrav_ns for x in dotmap_list]
    res.append(["relative_veh_hours", np.mean(data), np.std(data),
                np.nanpercentile(data, 5), np.nanpercentile(data, 95)])
    ax = sns.histplot(data, bins=bins)
    ax.axvline(30, ls=":")
    ax.set(xlabel=None, ylabel=None)
    plt.ylim(0, y_max)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.locator_params(axis='x', nbins=max_ticks)
    plt.savefig(topological_config.path_results + "figs/relative_veh_hours" + str_for_end + ".png")
    plt.close()

    pd.DataFrame({"var": [t[0] for t in res], "mean": [t[1] for t in res], "st_dev": [t[2] for t in res],
                  "P5": [t[3] for t in res], "P95": [t[4] for t in res]}) \
        .to_csv(topological_config.path_results + "kpis_means" + str_for_end + ".txt",
                sep=' ', index=False, header=False)


def probability_of_pooling_classes(dotmap_list, topological_config, name=None,
                                   _class_names=("C1", "C2", "C3", "C4"), max_ticks=5):
    sblts_exmas = topological_config.sblts_exmas
    if name is None:
        name = "per_class_prob_" + str(len(dotmap_list[0][sblts_exmas].requests))

    probs = {"c0": np.array([0, 0]), "c1": np.array([0, 0]), "c2": np.array([0, 0]), "c3": np.array([0, 0])}
    for rep in dotmap_list:
        df = rep['prob'].sampled_random_parameters.copy()
        df["VoT"] *= 3600
        df.set_index("new_index", inplace=True)
        c0 = df.loc[df["class"] == 0]
        c1 = df.loc[df["class"] == 1]
        c2 = df.loc[df["class"] == 2]
        c3 = df.loc[df["class"] == 3]

        schedule = rep[sblts_exmas].schedule
        non_shared = schedule.loc[schedule["kind"] == 1]
        a2 = frozenset(non_shared.index)

        probs["c0"] += np.array([len(a2.intersection(set(c0.index))), len(c0)])
        probs["c1"] += np.array([len(a2.intersection(set(c1.index))), len(c1)])
        probs["c2"] += np.array([len(a2.intersection(set(c2.index))), len(c2)])
        probs["c3"] += np.array([len(a2.intersection(set(c3.index))), len(c3)])

    x = _class_names

    def foo(i):
        return 100 * (probs["c" + str(i)][1] - probs["c" + str(i)][0]) / probs["c" + str(i)][1]

    y = [foo(t) for t in range(len(x))]

    ax = sns.barplot(data=pd.DataFrame({"names": x, "values": y}), x="names", y="values")
    ax.set(ylabel='Probability of sharing', xlabel=None)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.locator_params(axis='x', nbins=max_ticks)
    plt.savefig(topological_config.path_results + "figs/" + name + ".png")
    plt.close()


def amend_dotmap(dotmap, config):
    sblts_exmas = config.sblts_exmas
    requests = dotmap[sblts_exmas].requests[["id", "ttrav", "ttrav_sh", "u", "u_sh", "kind"]]
    schedule = dotmap[sblts_exmas].schedule
    probs = dotmap['prob'].sampled_random_parameters
    class_dict = {int(t[1]['new_index']): int(t[1]['class']) for t in list(probs.iterrows())}
    schedule["class"] = schedule["indexes"].apply(lambda x: [class_dict[t] for t in x])
    schedule["Veh_saved"] = (schedule['ttrav_ns'] - schedule['ttrav']) / schedule['ttrav_ns']
    requests = pd.merge(requests, probs[["class"]], left_on="id", right_index=True)
    return requests, schedule


def relative_travel_times_utility(df):
    df = df.assign(Relative_time_add=(df["ttrav_sh"] - df["ttrav"]))
    df['Relative_time_add'] = df['Relative_time_add'].apply(lambda x: 0 if abs(x) <= 1 else x)
    df['Relative_time_add'] = df['Relative_time_add'] / df['ttrav']
    df['Relative_utility_gain'] = (df['u'] - df['u_sh']) / df['u']
    return df


def separate_by_classes(list_dfs, standard=True):
    classes = dict()

    first_rep = True
    for rep in list_dfs:
        for class_no in [0, 1, 2, 3]:
            if first_rep:
                if standard:
                    classes["C" + str(class_no + 1)] = rep.loc[rep["class"] == class_no]
                else:
                    classes["C" + str(class_no + 1)] = rep.loc[[class_no in t for t in rep["class"]]]
            else:
                if standard:
                    classes["C" + str(class_no + 1)] = \
                        classes["C" + str(class_no + 1)].append(rep.loc[rep["class"] == class_no])
                else:
                    classes["C" + str(class_no + 1)] = \
                        classes["C" + str(class_no + 1)].append(rep.loc[[class_no in t for t in rep["class"]]])
        first_rep = False

    return classes


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def create_latex_output_df(df, column_format=None):
    if column_format is None:
        column_format = df.shape[1]*"c|" + "c"
    latex_df = df.to_latex(float_format="%.2f", column_format=column_format)
    latex_df = latex_df.replace("\\midrule", "\\hline")
    for rule in ["\\toprule", "\\bottomrule"]:
        latex_df = latex_df.replace(rule, "")
    return latex_df


def classes_analysis(dotmap_list, config, percentile=95, _bins=50, figsize=(4, 6), dpi=200):
    # Requests
    results_req = [amend_dotmap(indata, config)[0] for indata in dotmap_list]
    results_req = [relative_travel_times_utility(df) for df in results_req]
    size = len(results_req[0])
    classes_req = separate_by_classes(results_req)

    results_shared_req = [df.loc[df["kind"] > 1] for df in results_req]
    classes_shared_req = separate_by_classes(results_shared_req)

    # Schedule
    results_sch = [add_profitability(indata, config) for indata in dotmap_list]
    results_sch = [amend_dotmap(indata, config)[1] for indata in results_sch]
    classes_sch = separate_by_classes(results_sch, standard=False)

    results_shared_sch = [df.loc[df["kind"] > 1] for df in results_sch]
    classes_shared_sch = separate_by_classes(results_shared_sch, standard=False)

    variables = ["Relative_time_add", "Relative_utility_gain", "Profitability", "Veh_saved"]

    for var, sharing in itertools.product(variables, ['shared', 'all']):
        if var in ["Relative_time_add", "Relative_utility_gain"]:
            if sharing == 'shared':
                classes_dict = classes_shared_req
                res = results_shared_req
            else:
                classes_dict = classes_req
                res = results_req
            datasets = [t[var].apply(lambda x: x if x >= 0 else 0) for t in [v for k, v in classes_dict.items()]]
            whole_dataset = [j for i in [t[var].apply(lambda x: x if x >= 0 else 0) for t in res] for j in i]
        else:
            if sharing == 'shared':
                classes_dict = classes_shared_sch
                res = results_shared_sch
            else:
                classes_dict = classes_sch
                res = results_sch
            datasets = [t[var] for t in [v for k, v in classes_dict.items()]]
            whole_dataset = [j for i in [t[var] for t in res] for j in i]

        labels = [k for k, v in classes_dict.items()]
        maximal_percentile = np.nanpercentile(pd.concat(res, axis=0)[var], percentile)
        xlim_end = np.nanpercentile(pd.concat(res, axis=0)[var], 99.5)

        datasets = [whole_dataset] + datasets
        labels = ["All"] + labels

        fig, ax = plt.subplots(figsize=figsize)
        _line_styles = ['-', ':', '--', '-.', (0, (3, 5, 1, 5, 1, 5))]
        cmap = mpl.cm.get_cmap("tab10").colors
        for data, line_type, label, color in zip(datasets, _line_styles, labels, cmap):
            lw = 1.2 if label == "All" else 1
            plt.hist(data, density=True, histtype='step', label=label, cumulative=True, bins=len(data),
                     ls=line_type, lw=lw, color=color)
        # plt.hist(datasets, density=True, histtype='step', label=labels, cumulative=True, bins=_bins)
        # ax.axvline(x=maximal_percentile, color='black', ls=':', label='95%', lw=1)
        fix_hist_step_vertical_line_at_end(ax)
        if var != "Veh_saved":
            ax.set(xlabel=None, ylabel=None, yticklabels=[])
        else:
            ax.set(xlabel=None, ylabel=None)
        plt.xlim(left=-0.05 if var != "Profitability" else None, right=xlim_end)
        if var == "Profitability":
            # plt.legend([plt.Line2D(0, 0) for j in ax.get_legend_handles_labels()],loc="lower right")
            custom_lines = [Line2D([0], [0], color=color, lw=1, ls=_ls) for color, _ls in zip(cmap, _line_styles)]

            plt.legend(custom_lines, labels,  loc="lower right", fontsize=15)
        plt.savefig(config.path_results + "figs/" + "cdf_class_" + var + "_" + sharing + "_" + str(size) + ".png", dpi=dpi)
        plt.close()

        means = [np.mean(t) for t in datasets]
        st_devs = [np.std(t) for t in datasets]
        percentiles = [(np.nanpercentile(t, 75), np.nanpercentile(t, 90), np.nanpercentile(t, 95)) for t in datasets]
        df = pd.DataFrame({"Means": means, "St_dev": st_devs, "Q3": [t[0] for t in percentiles],
                           "90": [t[1] for t in percentiles], "95": [t[2] for t in percentiles]})
        df.index = ["All", "C1", "C2", "C3", "C4"]

        with open(config.path_results + 'per_class_' + var + "_" + sharing + "_" + str(size) + ".txt", "w") as file:
            file.write(create_latex_output_df(df, "c|c|c|c|c|c"))

    x = 0


def aggregated_analysis(dotmaps_list, config):
    sblts_exmas = config.sblts_exmas

    prob = []
    times = []

    list_len = len(dotmaps_list[0][sblts_exmas].requests)

    for rep in dotmaps_list:
        schedule = rep[sblts_exmas].schedule
        results = rep[sblts_exmas].res
        prob.append(list_len - len(schedule.loc[schedule["kind"] == 1]))
        times.append([results["PassHourTrav"], results["PassHourTrav_ns"]])

    relative_times = [(x - y) / y for x, y in times]

    prob_list = [t / list_len for t in prob]

    output_prob = np.round_((np.mean(prob_list), np.std(prob_list)), 4)
    output_times = np.round_((np.mean(relative_times), np.std(relative_times)), 6)
    print(f"number of requestes is: {list_len}")
    print(f"Pooling probability (mean, st.dev.): {output_prob}")
    print(f"Relative time extension (mean, st.dev.): {output_times}")

    return output_prob, output_times


def add_profitability(dotmap_data, config, speed=6, sharing_discount=0.3):
    speed = config.get('avg_speed', 6)
    sharing_discount = config.get('shared_discount', 0.3)
    sblts_exmas = config.sblts_exmas

    data = dotmap_data[sblts_exmas]['schedule'].loc[dotmap_data[sblts_exmas]['schedule']["kind"] > 1].copy()

    # data['dist_ns'] = data['ttrav_ns'] * speed
    # data['dist'] = data['ttrav'] * speed
    # data['profitability'] = data['dist_ns'] * (1 - sharing_discount) / data['dist']
    data['Profitability'] = data['ttrav_ns'] * (1 - sharing_discount) / data['ttrav']

    dotmap_data[sblts_exmas]['schedule']['Profitability'] = 1
    dotmap_data[sblts_exmas]['schedule']['dist'] = dotmap_data[sblts_exmas]['schedule']['ttrav'] * speed
    dotmap_data[sblts_exmas]['schedule'].loc[dotmap_data[sblts_exmas]['schedule']["kind"] > 1, "Profitability"] = \
        data['Profitability'].values

    return dotmap_data


def analyse_profitability(dotmaps_list, config, shared_all='all', speed=6, sharing_discount=0.3, bins=20, y_max=20,
                          save_results=True):
    sblts_exmas = config.sblts_exmas
    size = len(dotmaps_list[0][sblts_exmas].requests)
    speed = config.get('avg_speed', speed)
    sharing_discount = config.get('shared_discount', sharing_discount)

    relative_perspective = []

    for rep in dotmaps_list:
        discounted_distance = sum(rep[sblts_exmas].requests.loc[rep[sblts_exmas].requests["kind"] > 1]["dist"])
        veh_time_saved = rep[sblts_exmas].res["VehHourTrav_ns"] - rep[sblts_exmas].res["VehHourTrav"]
        veh_distance_on_reduction = discounted_distance - veh_time_saved * speed

        ns_dist = sum(rep[sblts_exmas].requests.loc[rep[sblts_exmas].requests["kind"] == 1]["dist"])

        # basic_relation = sum(rep[sblts_exmas].requests["dist"])/(rep[sblts_exmas].res["VehHourTrav_ns"]*speed)

        if shared_all == 'shared':
            profit_relation = discounted_distance * (1 - sharing_discount) / veh_distance_on_reduction
        elif shared_all == 'all':
            profit_relation = (discounted_distance * (1 - sharing_discount) + ns_dist) / (
                    veh_distance_on_reduction + ns_dist)
        else:
            raise Exception('Incorrect parameter "shared_all" it should be either "shared" or "all"')
        # relative_perspective.append(shared_relation/basic_relation)
        relative_perspective.append(profit_relation)

    if save_results:
        pd.DataFrame({"var": "Profitability", "mean": np.mean(relative_perspective), "st_dev": np.std(relative_perspective),
                      "P5": np.nanpercentile(relative_perspective, 5), "P95": np.nanpercentile(relative_perspective, 95)},
                     index=[0]).to_csv(config.path_results + "profitability_mean_" + str(size) + ".txt",
                                       sep=' ', index=False, header=False)

        ax = sns.histplot(relative_perspective, bins=bins)
        ax.axvline(1.097, ls=":")
        ax.set(xlabel=None, ylabel=None, yticklabels=[])
        plt.ylim(0, y_max)
        plt.savefig(config.path_results + "figs/" + "profitability_sharing_" + str(size) + ".png")
        plt.close()
    else:
        return relative_perspective


def individual_analysis(dotmap_list, config, no_elements=None, s=10):
    sblts_exmas = 'exmas'
    size = len(dotmap_list[0][sblts_exmas].requests)
    if no_elements is None:
        datasets = dotmap_list.copy()
    else:
        datasets = dotmap_list[:no_elements].copy()

    datasets = [d['exmas']['requests'].merge(d['prob']['sampled_random_parameters']['class'],
                                             left_on='id', right_index=True) for d in datasets]

    data = [t.loc[t['kind'] > 1] for t in datasets]
    agg_data = pd.concat(data)
    agg_data = relative_travel_times_utility(agg_data)
    agg_data['Relative_utility_gain'] = agg_data['Relative_utility_gain'].apply(lambda x: x if x >= 0 else abs(x))
    agg_data['VoT'] = agg_data['VoT'] * 3600
    agg_data.rename(columns={"class": "Class"}, inplace=True)
    agg_data["Class"] = agg_data["Class"] + 1
    agg_data["Class"] = agg_data["Class"].apply(lambda x: "C" + str(x))
    dict_labels = {'Relative_utility_gain': "$\mathcal{U}_r$", "Relative_time_add": "$\mathcal{T}_r$"}
    for y_var, x_var in itertools.product(['Relative_utility_gain', 'Relative_time_add'], ['VoT', 'WtS']):
        # palette = {0: 'green', 1: "orange", 2: "blue", 3: "red"}
        palette = {"C1": 'green', "C2": "orange", "C3": "blue", "C4": "red"}
        ax = sns.scatterplot(data=agg_data, x=x_var, y=y_var, hue="Class", palette=palette, s=s)
        ax.set(xlabel=None, ylabel=dict_labels[y_var])
        if y_var == "Relative_utility_gain" and x_var == "WtS":
            # plt.legend(labels=["C1", "C2", "C3", "C4"])
            # for handle in lgnd.legendHandles:
            #     handle.set_sizes([6.0])
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [1, 0, 2, 3]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        else:
            ax.get_legend().remove()
        ax.set_ylim(bottom=0)
        plt.savefig(config.path_results + "figs/" + x_var + '_' + y_var + "_" + str(size) + ".png")
        plt.close()


def individual_rides_profitability(dotmap_list, config, s=20, dpi=200):
    sblts_exmas = 'exmas'
    size = len(dotmap_list[0][sblts_exmas].requests)
    for rep in dotmap_list:
        rep = add_profitability(rep, config)

    dataset = pd.concat([t[sblts_exmas]['schedule'].loc[t[sblts_exmas]['schedule']["kind"] > 1] for t in dotmap_list])
    dataset.reset_index(drop=True, inplace=True)
    dataset.rename(columns={'degree': 'Degree'}, inplace=True)
    dataset["Distance"] = dataset['dist']/1000

    ax = sns.scatterplot(x=dataset['Distance'], y=dataset['Profitability'], hue=dataset['Degree'], s=s,
                         palette=sns.color_palette("tab10", max(dataset['Degree']) - 1))
    ax.set(xlabel=None, ylabel=None)
    # ax.set(xlabel=None, ylabel=None, yticklabels=[])
    # plt.ylim(0.6, 2.2)
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., title='Degree')
    plt.savefig(config.path_results + "figs/" + 'individual_rides_profitability_' + str(size) + ".png", dpi=dpi)
    plt.close()


def mixed_datasets_kpi(var, config, date, name0, name1, name2, graph_all=True, graph_type='kde'):
    sblts_exmas = config.sblts_exmas
    if type(var) == str:
        assert var in ['profit', 'veh', 'utility', 'pass'], "wrong var"
    else:
        assert type(var) == list, "wrong var"

    config.path_results0 = 'data/results/' + date + name0 + '/'
    config.path_results1 = 'data/results/' + date + name1 + '/'
    config.path_results2 = 'data/results/' + date + name2 + '/'

    if type(var) == list:
        vars = var
    else:
        vars = [var]

    if graph_all:
        rep_graphs0, dotmap_list0, all_graphs_list0 = load_data(config, config.path_results0)

    rep_graphs1, dotmap_list1, all_graphs_list1 = load_data(config, config.path_results1)
    rep_graphs2, dotmap_list2, all_graphs_list2 = load_data(config, config.path_results2)

    for var in vars:

        if var == 'profit':
            config0path = "profitability_mean_147.txt"
            config0path_num = 0
        else:
            config0path = "kpis_means_147.txt"
            multiplier = 100

        if var == 'veh':
            config0path_num = 2
            var1 = "VehHourTrav_ns"
            var2 = "VehHourTrav"
            var3 = var1

        elif var == 'utility':
            config0path_num = 0
            var1 = "PassUtility_ns"
            var2 = "PassUtility"
            var3 = var1

        elif var == 'pass':
            config0path_num = 1
            var1 = "PassHourTrav"
            var2 = "PassHourTrav_ns"
            var3 = var2

        else:
            pass

        if var == 'pass':
            _lw = 2
        else:
            _lw = 1.5

        with open(config.path_results0 + config0path, "r") as file:
            hline_val = file.readlines()

        h_line_value = float(hline_val[config0path_num].split()[1])

        if var in ['veh', 'utility', 'pass']:
            if graph_all:
                data0 = [multiplier * (x[sblts_exmas].res[var1] - x[sblts_exmas].res[var2]) / x[
                    sblts_exmas].res[var3] for x in dotmap_list0]
            data1 = [multiplier * (x[sblts_exmas].res[var1] - x[sblts_exmas].res[var2]) / x[
                sblts_exmas].res[var3] for x in dotmap_list1]
            data2 = [multiplier * (x[sblts_exmas].res[var1] - x[sblts_exmas].res[var2]) / x[
                sblts_exmas].res[var3] for x in dotmap_list2]
        else:
            if graph_all:
                data0 = analyse_profitability(dotmap_list0, config, save_results=False)
            data1 = analyse_profitability(dotmap_list1, config, save_results=False)
            data2 = analyse_profitability(dotmap_list2, config, save_results=False)

        if graph_all:
            data = pd.concat(axis=0, ignore_index=True, objs=[
                pd.DataFrame.from_dict({'value': data0, 'Demand size': 'baseline'}),
                pd.DataFrame.from_dict({'value': data1, 'Demand size': 'small'}),
                pd.DataFrame.from_dict({'value': data2, 'Demand size': 'big'})
            ])
        else:
            data = pd.concat(axis=0, ignore_index=True, objs=[
                pd.DataFrame.from_dict({'value': data1, 'Demand size': 'small'}),
                pd.DataFrame.from_dict({'value': data2, 'Demand size': 'big'})
            ])

        if var == "profit":
            plt.figure(figsize=(4.7, 3))
        else:
            plt.figure(figsize=(3.5, 3))

        data2 = pd.DataFrame()
        data2["value"] = data["value"]
        data2["Demand size"] = data["Demand size"]

        if graph_type == "kde":
            ax = sns.kdeplot(data=data2, x='value', hue='Demand size', palette=['red', 'blue', 'green'],
                             common_norm=False,
                             fill=True, alpha=.5, linewidth=0)
        elif graph_type == "hist":
            ax = sns.histplot(data, x='value', hue='Demand size', bins=20,
                              palette=['red', 'blue'])  # multiple="dodge", shrink=.8
        else:
            raise Exception("currently not implemented, graph type should be either 'kde' or 'hist'")
        ax.set(xlabel=None, ylabel=None, yticklabels=[])
        if not graph_all:
            ax.axvline(x=h_line_value, color='black', ls='--', label='Baseline mean', lw=_lw)
        plt.locator_params(axis='x', nbins=5)

        if var != "profit":
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            ax.get_legend().remove()
        else:
            if graph_all:
                legend = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", labels=['Dense', 'Sparse', 'Baseline'],
                                    title='Demand', borderaxespad=0)
            else:
                legend = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", labels=['Baseline mean', 'Sparse', 'Dense'],
                                    title='Demand', borderaxespad=0)

        plt.tight_layout()
        plt.savefig(config.path_results0 + "figs/mixed_" + var + ".png", dpi=200)
        plt.close()


def visualize_two_shareability_graphs(g1, g2, config, spec_name="shareability", dpi=200, figsize=(6, 6), edge_width=1,
                                      alpha_diff=0.6, thicker_common=1.5, alpha_common=0.7, only_netwulf=False):
    """
    Function designed to produce a graph presenting combination of two graphs with the same nodes
    @param dpi:
    @param figsize:
    @param edge_width:
    @param alpha_common:
    @param thicker_common:
    @param alpha_diff:
    @param config: config with run charactersitcs, especially including path-results
    @param spec_name: used in naming the output
    @param g1: graph
    @param g2: graph with the same nodes as g1
    @return: saved figure
    """
    if not only_netwulf:
        stylized_network, netwulf_config = nw.visualize(g1)
        layout = {i: nw.tools.node_pos(stylized_network, i) for i in range(len(list(g1.nodes)))}
        plt.figure(1, figsize=figsize, dpi=dpi)
        ax = nx.draw_networkx_nodes(g1, pos=layout, node_color="black", node_size=20)
        edges1 = set(g1.edges)
        edges2 = set(g2.edges)
        nx.draw_networkx_edges(g1, pos=layout, edgelist=edges1.intersection(edges2), width=thicker_common*edge_width, edge_color="red", alpha=alpha_common)
        nx.draw_networkx_edges(g1, pos=layout, edgelist=edges1.difference(edges2), width=edge_width, edge_color="blue", alpha=alpha_diff)
        nx.draw_networkx_edges(g2, pos=layout, edgelist=edges2.difference(edges1), width=edge_width, edge_color="green", alpha=alpha_diff)
        plt.box(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False)
        plt.savefig(config.path_results + "figs/mixed_" + spec_name + "_graph.png", transparent=True, pad_inches=0)
        plt.close()
    else:
        edges1 = set(g1.edges)
        edges2 = set(g2.edges)
        edges0 = edges1.intersection(edges2)
        edges_o1 = edges1.difference(edges2)
        edges_o2 = edges2.difference(edges1)
        G = nx.Graph()
        G.add_weighted_edges_from([(x, y, 2) for x, y in edges0])
        G.add_weighted_edges_from([(x, y, 1) for x, y in edges_o1])
        G.add_weighted_edges_from([(x, y, 1) for x, y in edges_o2])
        visualize(G)


