import sys

import datasets
from analysis.RDT_analysis import run_analysis


LPT = [(1, 2), (2, 4), (4, 8), (8, 16), (32, 64), (1, 5), (1, 10), (3, 6)]  # leaf parameter tuples

# all prediction and aggregation methods we implemented and which aggregation can be used on which decision tree method
DT_MODES = ["probability", "linear", "beta_1_1", "ucb", "fast_cb", "instances", "raw_beta_dist", "plaus_values_4"]
AGG_MODES = {"probability": ["average", "maximum", "median", "voting", "gm_average", "gm_maximum", "gm_median"]}
AGG_MODES.update({"linear": ["average", "maximum", "median", "voting", "gm_average", "gm_maximum", "gm_median"]})
AGG_MODES.update({"beta_1_1": ["average", "maximum", "median", "voting", "gm_average", "gm_maximum", "gm_median"]})
AGG_MODES.update({"ucb": ["average", "maximum", "median", "voting", "gm_average", "gm_maximum", "gm_median"]})
AGG_MODES.update({"fast_cb": ["average", "maximum", "median", "voting", "gm_average", "gm_maximum", "gm_median"]})
AGG_MODES.update({"instances": ["pooling", "evidence"]})
AGG_MODES.update({"raw_beta_dist": ["beta_agg_sum", "beta_agg_max"]})
AGG_MODES.update({"plaus_values_4": ["average", "maximum", "median", "voting", "gm_average", "gm_maximum",
                                     "gm_median", "dempster", "cautious", "epistemic_weight"]})


def main():
    """ Run the main analysis comparing the prediction and aggregation methods."""
    num_arguments = len(sys.argv) - 1
    if num_arguments == 1 and sys.argv[1] == "test":
        print("Python code test successful")
        return
    elif num_arguments != 2:
        raise Exception("Two arguments (start index, end index) need to be specified!")
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
    lpt = LPT
    dt_modes = DT_MODES
    agg_modes = AGG_MODES
    for data_ident in datasets.ALL[start_index:end_index]:
        run_analysis(data_ident, 0.5, [3], 5, lpt, 100, dt_modes, agg_modes)
        print(f"{data_ident} finished")


if __name__ == "__main__":
    main()
