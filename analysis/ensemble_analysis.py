import numpy as np
import matplotlib.pyplot as plt

import datasets
from random_tree import RandomDecisionTree, check_data
from tree_ensemble import RDT


def method_ident_to_name(dt_ident, agg_ident, short_names=False):
    """ Return a better name which is more readable than the respective identifier. For some identifiers exists a short
    method name which can also be chosen. """
    if dt_ident == "instances":
        if agg_ident == "pooling":
            return "Pooling"
        elif agg_ident == "evidence":
            return "Evidence Accumulation"
        else:
            raise ValueError(f"There is no {agg_ident} aggregation method for {dt_ident}")
    elif dt_ident == "dt_baseline":
        return "Single Decision Tree Reference"
    elif dt_ident == "rf_baseline":
        return "Random Forest Reference"
    dt_name = {"probability": "Probability",
               "fast_cb": "Confidence Bounds",
               "plaus_values_4": "Plausibility",
               "linear": "Linear Prediction",
               "beta_1_1": "Laplace",
               "ucb": "UCB inspired",
               "raw_beta_dist": "Beta Distributions"}
    agg_name = {"average": "Average",
                "maximum": "Maximum",
                "median": "Median",
                "voting": "Voting",
                "gm_average": "GM-Average",
                "gm_maximum": "GM-Maximum",
                "gm_median": "GM-Median",
                "dempster": "Dempster",
                "cautious": "Cautious",
                "epistemic_weight": "Eps-Weight",
                "beta_agg_sum": "Sum",
                "beta_agg_max": "Max"}
    if short_names:
        if dt_ident == "probability":
            if agg_ident == "average":
                return "Average Probability"
            elif agg_ident == "voting":
                return "Voting"
            else:
                return dt_name[dt_ident]
        elif agg_ident == "average":
            return dt_name[dt_ident]
        elif dt_ident == "plaus_values_4" and (agg_ident == "dempster" or agg_ident == "cautious"):
            return agg_name[agg_ident]
    return dt_name[dt_ident] + " : " + agg_name[agg_ident]


def data_ident_to_name(data_ident):
    """ Return the name of the dataset as used in the paper instead of the dataset identifier used in the code. """
    ident_to_name = {"breast_cancer": "breast-cancer", "magic_telescope": "telescope", "arff_diabetes": "diabetes",
                     "arff_sonar": "sonar", "arff_ionosphere": "ionosphere", "arff_spambase": "spambase",
                     "transfusion": "transfusion", "pop_failures": "climate", "biodeg": "biodeg",
                     "banknote_authentication": "banknote", "scene": "scene", "Skin_NonSkin": "skin",
                     "webdata": "webdata", "MiniBooNE_PID": "particle", "vehicle_sens": "vehicle",
                     "arff_tic-tac-toe": "tic-tac-toe", "arff_TrainingDataset": "phishing", "arff_vote": "voting",
                     "arff_elecNormNew": "electricity", "arff_airlines": "airlines", "mushroom": "mushroom"}
    return ident_to_name[data_ident]


def apms_ensemble(data_ident, test_size, random_state, min_samples_list, design_info, ensemble_size=100, save=False,
                  legend=True, log=False, save_name="apms_test", data_given=False,
                  ens_methods=[("probability", "average")]):
    """ The APMS (short for accuracy per min samples) function creates a plot for the given parameters. Here, the
    minimum number of instances per leaf parameter is plotted on the x-axis, the accuracy on the y-axis. Each dot
    represents one single random tree with a specific accuracy on a separate test set. In addition to that, a dotted
    line shows the average of this single trees and some opaque ares show the standard deviation (times 1, 2, and 3).
    Lastly, the methods in the ens_methods parameter can be used to also plot lines for the performance of an ensemble
    on the same trees. The main purpose of this function is to see how a specific dataset performs depending on its
    minimum leaf size, how much this can vary from tree to tree (randomly) and how much certain methods improve upon
    that. """
    # calculate the data needed for the plots
    if not data_given:
        x_axis = []
        y_data = np.zeros((ensemble_size, len(min_samples_list)))
        y_ens = {}
        for dt_method, agg_method in ens_methods:
            y_ens[(dt_method, agg_method)] = []
        # load the dataset
        x_train, y_train, x_test, y_test = datasets.load_train_test(data_ident, test_size, random_state)
        # for every min_samples parameter
        for count, min_sample in enumerate(min_samples_list):
            # create and fit the ensemble including all single trees
            test_rdt = RDT(number_trees=ensemble_size, min_samples_leaf=min_sample, random_state=count)
            test_rdt.fit(x_train, y_train)
            x_test, y_test, _, _ = check_data(x_test, y_test)
            # calculate the accuracy for the different ensemble methods
            for dt_method, agg_method in ens_methods:
                res = test_rdt.predict(x_test, dt_method, agg_method, empty_leafs=3)
                acc = np.average(res[:, 0] == y_test)
                y_ens[(dt_method, agg_method)].append(acc)
            x_axis += [min_sample]
            # calculate the accuracy on the single trees
            for i, tree in enumerate(test_rdt.trees):
                res = tree.predict_class(x_test, empty_leafs=3, verbose=False)
                acc = np.average(res == y_test)
                y_data[i, count] = acc
            # show that there is some progress after every min_sample step
            print(str(100*(count+1)/len(min_samples_list))[:4] + "%")
        # save the data so it can be loaded again later (much faster than calculating)
        np.save("analysis/data/" + data_ident + "_data", y_data)
        for dt_method, agg_method in ens_methods:
            y_ens[(dt_method, agg_method)] = np.array(y_ens[(dt_method, agg_method)])
            np.save("analysis/data/" + data_ident + "_" + dt_method + "_" + agg_method, y_ens[(dt_method, agg_method)])
    # load the data needed for the plots, can save much time if you just want smaller changes on the visualization
    else:
        print("Loading data from disk.")
        x_axis = min_samples_list
        y_data = np.load("analysis/data/" + data_ident + "_data.npy")
        y_ens = {}
        for dt_method, agg_method in ens_methods:
            y_ens[(dt_method, agg_method)] = np.load("analysis/data/" + data_ident + "_" + dt_method + "_" + agg_method
                                                     + ".npy")
    # standard deviation
    y_avg = np.average(y_data, axis=0)
    y_std = np.std(y_data, axis=0)
    plt.fill_between(x_axis, np.maximum(0, y_avg - 3 * y_std), np.minimum(1, y_avg + 3 * y_std), alpha=0.3, color="b")
    plt.fill_between(x_axis, np.maximum(0, y_avg - 2 * y_std), np.minimum(1, y_avg + 2 * y_std), alpha=0.3, color="b")
    plt.fill_between(x_axis, np.maximum(0, y_avg - 1 * y_std), np.minimum(1, y_avg + 1 * y_std), alpha=0.3, color="b")
    for i in range(ensemble_size):
        # single trees
        plt.scatter(x_axis, y_data[i, :], c="c", s=30, alpha=0.4, label="Single Tree Accuracies")

    method_list = []
    for (method, _, _) in design_info:
        method_list.append(method)

    for count, (dt_method, agg_method) in enumerate(ens_methods):
        # ensemble methods
        zorder = method_list.index((dt_method, agg_method))
        color = design_info[zorder][1]
        linestyle = design_info[zorder][2]
        label = method_ident_to_name(dt_method, agg_method, short_names=True)
        plt.plot(x_axis, y_ens[(dt_method, agg_method)], label=label, c=color, lw=2, linestyle=linestyle)
    if log:
        plt.gca().set_xscale("log", base=2)
        plt.xticks([1, 2, 4, 8, 16, 32, 64, 128])
    # average of single tree predictions
    plt.plot(x_axis, np.average(y_data, axis=0), c="b", lw=2, label="Average Tree Accuracy", linestyle="dashed")
    plt.xlabel("Minimum Leaf Size", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.title(data_ident_to_name(data_ident), fontsize=22)
    plt.grid(True)

    plt.tick_params(axis="y", which="both", left=True, right=True, labelleft=True, labelright=True, labelsize=15)
    plt.tick_params(axis="x", which="minor", bottom=False)
    plt.tick_params(axis="x", which="major", bottom=True, labelsize=15)

    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

    by_label = None
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1, 1))

    if save:
        plt.savefig("analysis/data/" + save_name + ".pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()

    if legend:
        # also plot the legends in a seperate file (two versions, all as separate pdfs)
        legend_fig = plt.figure("Legend plot")
        legend_fig.legend(by_label.values(), by_label.keys(), loc="center", ncol=11, frameon=False)
        legend_fig.savefig(f"analysis/data/legend_one_col.pdf", bbox_inches="tight", pad_inches=0)
        legend_fig.clf()

        legend_fig = plt.figure("Legend plot")
        legend_fig.legend(list(by_label.values())[:5], list(by_label.keys())[:5], loc="center", ncol=11, frameon=False)
        legend_fig.savefig(f"analysis/data/legend_two_col_A.pdf", bbox_inches="tight", pad_inches=0)
        legend_fig = plt.figure("Legend plot")
        legend_fig.clf()
        legend_fig.legend(list(by_label.values())[5:], list(by_label.keys())[5:], loc="center", ncol=11, frameon=False)
        legend_fig.savefig(f"analysis/data/legend_two_col_B.pdf", bbox_inches="tight", pad_inches=0)
        legend_fig.clf()


def go():
    # parameters for the graphs shown in the paper
    for data_ident in ["magic_telescope", "arff_ionosphere", "webdata", "arff_airlines"]:
        methods = [("probability", "average"), ("fast_cb", "average"), ("plaus_values_4", "average"),
                   ("instances", "pooling"), ("instances", "evidence")]
        design_info = [(("probability", "average"), "tab:red", "dashed"),
                       (("fast_cb", "average"), "tab:green", "solid"),
                       (("plaus_values_4", "average"), "tab:cyan", "solid"),
                       (("instances", "pooling"), "tab:brown", "solid"),
                       (("instances", "evidence"), "tab:blue", "solid")]
        apms_ensemble(data_ident, 0.33, 42, list(range(1, 13)) + [14, 16, 32, 64, 128], design_info, 100, True,
                      legend=True, log=True, save_name="HEUTE_tree_sim_paper_" + data_ident, data_given=False,
                      ens_methods=methods)
        print(f"{data_ident} finished")
