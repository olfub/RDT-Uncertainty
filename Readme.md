# Combining Predictions under Uncertainty: The Case of Random Decision Trees



This is the repository for the paper "Combining Predictions under Uncertainty: The Case of Random Decision Trees".



## Setup

This project is written in Python 3.7 and uses the packages as specified in requirements.txt

To run the code, Python and these packages need to be installed and the data needs to be downloaded and stored as specified below.



## Data

This paper considers several binary classification datasets with different sizes.

The data is collected from different sources. You can also choose to only use a subset of these datasets. The following table shows the sources but the tree datasets from OpenML do not need to be downloaded by hand as this is done within the code using the scikit-learn function "fetch_openml". The breast-cancer dataset is simply loaded from scikit-learn.

All datasets with an .arff ending should be stored in the "data_sets/arff" folder so that they can be read by the same function. The other datasets (which are not loaded by downloading within the code) should be stored in the "data_sets/UCI" folder.

| Name          | Source                                                       |
| ------------- | ------------------------------------------------------------ |
| scene         | https://www.openml.org/d/312                                 |
| webdata       | https://www.openml.org/d/350                                 |
| transfusion   | https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center |
| biodeg        | http://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation   |
| telescope     | https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope |
| diabetes      | https://github.com/renatopp/arff-datasets                    |
| voting        | https://github.com/renatopp/arff-datasets                    |
| spambase      | https://github.com/renatopp/arff-datasets                    |
| electricity   | https://moa.cms.waikato.ac.nz/datasets/                      |
| banknote      | https://archive.ics.uci.edu/ml/datasets/banknote+authentication |
| airlines      | https://moa.cms.waikato.ac.nz/datasets/                      |
| sonar         | https://github.com/renatopp/arff-datasets                    |
| mushroom      | https://archive.ics.uci.edu/ml/datasets/Mushroom             |
| vehicle       | https://www.openml.org/d/357                                 |
| phishing      | https://archive.ics.uci.edu/ml/datasets/Phishing+Websites    |
| breast-cancer | Loaded from scikit-learn ("load_breast_cancer")              |
| ionosphere    | https://github.com/renatopp/arff-datasets                    |
| tic-tac-toe   | https://github.com/renatopp/arff-datasets                    |
| particle      | https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification |
| skin          | https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation    |
| climate       | https://archive.ics.uci.edu/ml/datasets/Climate+Model+Simulation+Crashes |

## Experiments

Figure 5 of the paper can be generated by running create_visualizations.py.

Running run_analysis.py generates the data for the main experiment.

Please note, that running the full experiment in run_analysis.py can take a significant amount of time and storage for the larger datasets (several days for all datasets on our machines). Just running parts of the experiment (smaller datasets or leaving out some methods) can shorten this time by a lot.

## Further Notes

This code contains some more methods than introduced in the paper. However, we chose to not include these in the paper because they did not perform notably well, were not particularly interesting and we wanted to keep the paper concise. However, if there is interest, these method can be found within the code.

If there is confusion because the code sometimes uses slightly different identifiers for method names or dataset names, in "analysis/ensemble_analysis.py", there are functions which take these identifiers and output the names / descriptions used in the paper which should be more readable if there is any confusion.

## Citation

```
@InProceedings{fb:DS-21, 
    author="Busch, Florian Peter and Kulessa, Moritz and Loza Menc{\'i}a, Eneldo and Blockeel, Hendrik",
    title="Combining Predictions under Uncertainty: Random Decision Trees",
    booktitle="Proceedings of the 24th International Conference on Discovery Science",
    year="2021",
    note= "In press."
}
```