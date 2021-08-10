import numpy as np
import scipy

from abc import ABC, abstractmethod

from prediction import prediction_methods
from prediction.prediction_utility import prepare_empty_leafs_2, finish_empty_leafs_2

# instead of using "Numeric" and "Nominal" as feature types, 0 and 1 is used because it is more space efficient
FEATURE_TYPE = {0: "Numeric", 1: "Nominal"}


def is_number(s):
    """
    Test, whether the given input s (usually String) can be converted into a float (or int) or not and return True or
    False, respectively. Using the function for Strings only is recommended.
    :param s: the input value, usually a String
    :return: True if the input can be cast to a float, False otherwise
    """
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def check_data(x, y=None, suppress_info=False, class_converter=None):
    """
    Checks, whether the dataset (x, y) has a valid structure. If no y is given, only x is checked and returned. Also
    applies necessary preprocessing data transformations if x and y is given.
    :param x: feature array of the dataset
    :param y: labels of the dataset
    :param suppress_info: if False, additional information is printed in the console
    :param class_converter: if not None, uses this dictionary to apply the same class transformations on the current
    data as on the data, where that class_converter was set
    :return: tuple (x, y, len(classes), new_class_converter)
        WHERE
        x is the feature array of the dataset, which now is a numpy array (cast if necessary)
        y is a numpy array describing the labels of the dataset as integer numbers (labels starting at 0, then 1, 2,...)
        len(classes) is the number of classes in y
        new_class_converter is a dictionary which stores which original classes are transformed to which class numbers
    """
    # check, whether the structure of the dataset is valid
    if type(x) is not np.ndarray:
        if type(x) is scipy.sparse.csr.csr_matrix:
            x = x.toarray()
        else:
            raise TypeError("The data x must be a numpy array (np.ndarray)")
    if y is None:
        return x
    if type(y) is not np.ndarray:
        if type(y) is not list:
            raise TypeError("The data x must be a numpy array (np.ndarray) or a list")
        else:
            y = np.array(y)

    if len(y.shape) != 1:
        raise TypeError("The labels (y) must have only one dimension (for example no multilabel)")
    if len(x.shape) != 2:
        raise TypeError("The shape of the data x is invalid (not two-dimensional)")
    if x.shape[0] != y.shape[0]:
        raise TypeError("The number of instances of the data (x) does not match the number of labels (y)")

    # check the class types and change the values, if necessary
    new_class_converter = {}
    classes = np.unique(y)
    # initial look at the dataset
    if class_converter is None:
        # check, whether the classes have valid types (0, 1, 2, ...)
        valid_classes = True
        # classes are valid, if they are all numbers, ranging from 0 to the number of classes minus 1
        if not (issubclass(classes.dtype.type, np.integer)) and not (issubclass(classes.dtype.type, np.float)):
            valid_classes = False
        else:
            for i in range(len(classes)):
                if i not in classes:
                    valid_classes = False
                    break
        # replace with valid class types if necessary
        if not valid_classes:
            if not suppress_info:
                print("Changing class labels to allow for better calculation:")
            new_y = np.zeros_like(y, dtype=int)
            for new_class, old_class in enumerate(classes):
                new_class_converter[old_class] = new_class
                new_y[y == old_class] = new_class
                if not suppress_info:
                    print("Class \"" + str(old_class) + "\" is now \"" + str(new_class) + "\"")
            y = new_y
        else:
            for label in classes:
                new_class_converter[label] = label

    # use a existing conversion of classes (captures some special cases where you want the same conversion of the test
    # set which was also used for the train set)
    else:
        previous_classes = class_converter.keys()
        if not suppress_info:
            print("Changing class labels to allow for better calculation:")
        new_y = np.zeros_like(y, dtype=int)
        for prev_class in previous_classes:
            new_class = class_converter[prev_class]
            new_y[y == prev_class] = new_class
            if not suppress_info:
                print("Class \"" + str(prev_class) + "\" is now \"" + str(new_class) + "\"")
        unseen_classes = np.setdiff1d(classes, np.array(list(previous_classes)), assume_unique=True)
        new_class_converter = class_converter.copy()
        for count, unseen_class in enumerate(unseen_classes):
            new_class = len(class_converter.keys()) + count
            new_class_converter[unseen_class] = new_class
            new_y[y == unseen_class] = new_class
            if not suppress_info:
                print("Class \"" + str(unseen_class) + "\" is now \"" + str(new_class) + "\"")
        y = new_y

    y = y.astype(int)
    return x, y, len(classes), new_class_converter


def set_data_feature_information(x):
    """
    Given a dataset x (labels are not necessary), calculate and return the types of each feature attribute and the
    set of possible nominal feature values. This function uses the first element for the feature type calculation (types
    are required to be consistent throughout the dataset).
    :param x: input feature array
    :return: tuple (data_feature_types, nominal_feature_values)
        WHERE
        data_feature_types is a list, containing the types (nominal, numeric) of each feature vector
        nominal_feature_values is a dictionary, mapping the feature vector number on a list of possible values (only
        for nominal features, can be used to create nominal splits later)
    """
    data_feature_types = []
    nominal_feature_values = {}
    for i in range(x.shape[1]):
        # numeric
        if all(is_number(x[j][i]) for j in range(x.shape[0])):
            data_feature_types.append(0)
        # nominal
        else:
            data_feature_types.append(1)
            possible_feature_values = list(np.unique(x[:, i]))
            nominal_feature_values[i] = possible_feature_values
    return data_feature_types, nominal_feature_values


def set_nominal_feature_values(x, data_feature_types):
    """
    Given a dataset x (labels are not necessary), calculate and return the set of possible nominal feature values.
    Compared to set_data_feature_information, this function also takes in a predefined list of features attribute
    types.
    :param x: input feature array
    :param data_feature_types: list, containing the types (nominal, numeric) of each feature vector
    :return: dictionary, mapping the feature vector number on a list of possible values (only for nominal features, as
    defined by data_feature_types)
    """

    # check validity of data_feature_types
    if not (all(dft == 0 or dft == 1 for dft in data_feature_types)):
        raise TypeError("Any type must be either 0 or 1")
    if len(data_feature_types) != x.shape[1]:
        raise TypeError("Predefined data-feature-types do not fit the size of the data")

    # set nominal_feature_values
    nominal_feature_values = {}
    for i in range(x.shape[1]):
        if FEATURE_TYPE[data_feature_types[i]] == "Nominal":
            possible_feature_values = list(np.unique(x[:, i]))
            nominal_feature_values[i] = possible_feature_values
    return nominal_feature_values


def print_node(node):
    """ Print basic information about the given node in one line. """
    children_list = [child.node_id for child in node.children]
    print(f"ID: {str(node.node_id)}, Depth: {str(node.depth)}, Instances: {str(np.sum(node.classes))}, Children : "
          f"{str(children_list)}")


class Split(ABC):
    """
    Abstract class of a Split (of a node within a decision tree), a specific subclass is needed for the ability to
    process a split request.
    """

    @abstractmethod
    def __init__(self, split_index):
        """
        Initialize a split using a split-index, describing which feature the split will use.
        :param split_index: the feature index used for the split
        """
        self.split_index = split_index

    @abstractmethod
    def apply_split(self, x, y, min_samples_leaf, *args):
        """
        Apply a split.
        """
        pass

    @abstractmethod
    def move_to_next_node(self, new_x, feature_values):
        """
        Pass new_x to the next child node.
        """
        pass


class NumericalSplit(Split):
    """
    Class, which describes a numerical split. When splitting, this splits randomly chooses an instance and an operator
    (either "<" together with ">=" or "<=" together with ">") to apply a split. Note, that it is possible that 100% of
    all instances will be split to the same leaf and therefore another, empty leaf will be created.
    """

    def __init__(self, split_index):
        """
        Create an instance of the NumericalSplit. The split-index is set and the attributes operator and value are
        declared but not initialized.
        :param split_index: the feature index used for the split
        """
        super().__init__(split_index)
        self.operator = None  # 0: include value in the right child node, 1: include value in the left child node
        self.value = None  # the value used as the split value

    def as_string(self):
        """
        Returns a string representation, describing this split.
        """
        if self.operator == 0:
            comparison = "<"
        else:
            comparison = "<="
        return f"Numerical split, feature index: {self.split_index}, split: {comparison} {self.value}"

    def apply_split(self, x, y, min_samples_leaf, random_state=None, value=None, operator=None):
        """
        Apply a split of the instances (x, y) according to the specified parameters. If value or operator are None, they
        will be chosen in regards to the random_state.
        :param x: feature data
        :param y: data labels
        :param min_samples_leaf: min_samples_leaf criterion (minimum instance size of resulting leafs)
        :param random_state: random state, determining random behavior
        :param value: value used to split on
        :param operator: either includes value "<=" (left) or ">=" (right)
        :return: x and y divided into two parts (left and right child)
        """
        if random_state is None:
            return ValueError("random_state parameter needs to not be None")
        # set operator and value if any are None
        if operator is None:
            self.operator = random_state.randint(2)
        else:
            self.operator = operator
        if value is None:
            index = random_state.randint(x.shape[0])
            self.value = float(x[index, self.split_index])
        else:
            self.value = value

        # iterator over the instances (x, y)
        instance_iterator = zip(list(x), list(y))

        # apply the split by putting each instance (x, y) in a list for either the left or right instances
        if self.operator == 0:
            left_instances = [instance for instance in instance_iterator
                              if float(instance[0][self.split_index]) < self.value]
            instance_iterator = zip(list(x), list(y))  # new iterator
            right_instances = [instance for instance in instance_iterator
                               if float(instance[0][self.split_index]) >= self.value]
        elif self.operator == 1:
            left_instances = [instance for instance in instance_iterator
                              if float(instance[0][self.split_index]) <= self.value]
            instance_iterator = zip(list(x), list(y))  # new iterator
            right_instances = [instance for instance in instance_iterator
                               if float(instance[0][self.split_index]) > self.value]
        else:
            raise RuntimeError("Unexpected error in \"process_split_request\", error number: 1")

        # transform the list into the typical x, y form for each child
        left_x = np.array([instance[0] for instance in left_instances])
        left_y = np.array([instance[1] for instance in left_instances])
        right_x = np.array([instance[0] for instance in right_instances])
        right_y = np.array([instance[1] for instance in right_instances])

        # check, whether min_samples_leaf holds
        if len(left_x) < min_samples_leaf or len(right_x) < min_samples_leaf:
            return []

        # return the children as part of a list
        return [[left_x, left_y], [right_x, right_y]]

    def move_to_next_node(self, new_x, feature_values):
        """
        For each element in new_x, returns the index corresponding to the child node, where the element moves to. The
        feature values are not used here, but only within the NominalSplit class.
        :param new_x: instances for which the children (ids) are returned
        :param feature_values: unused parameter for this class
        :return: list of child ids, one id for each input instance
        """
        if self.operator == 0:
            child_ids = [0 if float(instance[self.split_index]) < self.value else 1 for instance in new_x]
        elif self.operator == 1:
            child_ids = [0 if float(instance[self.split_index]) <= self.value else 1 for instance in new_x]
        else:
            raise RuntimeError("Unexpected error in \"move_to_next_node\", error number: 1")
        return child_ids


class NominalSplit(Split):
    """
    Class, which describes a nominal split. When splitting, this splits uses the given feature index to apply a split.
    Note, that it is possible (and rather likely) that empty leafs will be created.
    """

    def __init__(self, split_index):
        """
        Create an instance of the NominalSplit and set the split-index.
        """
        super().__init__(split_index)

    def as_string(self):
        """
        Returns a string representation, describing this split.
        """
        return f"Nominal split, feature index: {self.split_index}"

    def apply_split(self, x, y, min_samples_leaf, feature_values=None):
        """
        Apply a split of the instances (x, y) according to the specified nominal features and the min_samples_leaf
        criterion.
        :param x: feature data
        :param y: data labels
        :param min_samples_leaf: min_samples_leaf criterion (minimum instance size of two largest resulting leafs)
        :param feature_values: possible feature values for that feature
        :return: x and y divided into several parts (one child for each feature value)
        """
        if feature_values is None:
            return ValueError("feature_values parameter needs to not be None")

        values = feature_values[self.split_index]
        # use a children dictionary, where the nominal feature maps to a list of instances (with that feature)
        children = {}
        for value in values:
            children[value] = []

        instance_iterator = zip(list(x), list(y))

        # assign the instances to corresponding children
        for instance in instance_iterator:
            children[instance[0][self.split_index]].append(instance)

        # create a children_list, where each element represents a child
        # each child contains the instances (x, y)
        children_list = []
        for value in values:
            child_x = np.array([instance[0] for instance in children[value]])
            child_y = np.array([instance[1] for instance in children[value]])
            children_list.append([child_x, child_y])

        instance_counts = sorted([len(child[0]) for child in children_list])

        # the two largest children must fulfill min_samples_leaf (design decision), therefore, checking the second
        # largest is enough
        if (min_samples_leaf > 0) and ((len(instance_counts) == 1) or instance_counts[-2] < min_samples_leaf):
            children_list = []

        return children_list

    def move_to_next_node(self, new_x, feature_values):
        """
        For each element in new_x, returns the index corresponding to the child node, where the element moves to. The
        feature_values are used so that the order of the possible feature values are known.
        :param new_x: instances for which the children (ids) are returned
        :param feature_values: possible feature values for that feature
        :return: list of child ids, one id for each input instance
        """
        values = feature_values[self.split_index]
        # return the id of an "imaginary" child in case of a new nominal feature
        # this child will then be treated as an empty node
        child_ids = [values.index(instance[self.split_index]) if instance[self.split_index] in values else len(values)
                     for instance in new_x]
        return child_ids


class Node:
    """
    A node within a decision tree.
    """

    def __init__(self, y, num_classes, depth, node_id):
        """
        Create a node by setting the id, the instances (x, y) and the depth of the node. Also declare uninitialized
        attribute: the list of children node (stays empty if the node has no children) and the split.
        :param y: classes of that node
        :param num_classes: number of classes of the entire dataset (node can have less classes)
        :param depth: depth of this node
        :param node_id: id of the node, unique within its tree
        """
        self.node_id = node_id
        self.classes = np.zeros(num_classes, dtype=int)
        for label in y:
            self.classes[label] += 1
        self.depth = depth
        self.children = []
        self.split = None  # split stays None if Node is a leaf

    def as_string(self):
        """
        Returns a string representation, describing this node (not unique, is not an id).
        """
        if self.is_leaf():
            first_descr = "Leaf node"
            child_str = ""
        else:
            first_descr = self.split.as_string()
            child_str = f", {len(self.children)} children"
        return f"{first_descr}, depth {self.depth}{child_str}, classes: {self.classes}"

    def is_leaf(self):
        """
        Returns whether this node is a leaf node.
        """
        return len(self.children) == 0

    def classes(self):
        """
        Returns the class distribution of that node.
        """
        return self.classes

    def process_split_request(self, x, y, max_depth, min_samples_split, min_samples_leaf, random_state, nominal_splits,
                              data_feature_types, nominal_feature_values, next_id, verbose):
        """
        When calling this method, this node is split if splitting this node does not violate any criteria. If, for any
        reason, no split is made, None is returned. If a split was successfully applied, a list of tuples is returned,
        where each tuple contains a child node and the list of nominal splits applied on this tree up until and
        including this node (so that that nominal feature will not be used another time).
        :param x: feature data
        :param y: data labels
        :param max_depth: maximum depth of the tree (parameter for RDT)
        :param min_samples_split: minimum number of instances to try to split (parameter for RDT)
        :param min_samples_leaf: minimum number of instances in a leaf node (parameter for RDT, exceptions exist)
        :param random_state: random state, can be used for deterministic behavior
        :param nominal_splits: splits on nominal features applied up until this point
        :param data_feature_types: types of each feature (nominal or numeric)
        :param nominal_feature_values: list of all possible values for each nominal feature
        :param next_id: id to be given to the next node
        :param verbose: if True, print additional information to the console
        :return: a list of tuples, one entry for each child, where the tuple contains the child, the nominal splits
        list, the x values of the respective child (feature data) and y of the respective child (labels)
        """

        # apply max_depth and min_samples_split criteria
        if len(x) < min_samples_split or (max_depth is not None and self.depth >= max_depth):
            return None

        # list, containing all possible splits indices for features
        possible_split_indices = list(range(len(data_feature_types)))

        # create a new copy of this list to avoid changing this list for siblings and their children
        new_nominal_splits = nominal_splits.copy()

        # try to apply a split which upholds the min_samples_leaf criterion
        trying_splits = True
        number_tries = 0
        children_list = []
        while trying_splits:

            # filter out past nominal splits
            possible_split_indices = [index for index in possible_split_indices if index not in new_nominal_splits]
            # return, if it is impossible to further split the data
            if len(possible_split_indices) == 0:
                return None

            # randomly choose a index to be used for the split
            split_index = random_state.randint(len(possible_split_indices))
            split_index = possible_split_indices[split_index]
            if verbose:
                print(self.node_id, possible_split_indices, split_index)

            # apply the split
            if FEATURE_TYPE[data_feature_types[split_index]] == "Numeric":
                self.split = NumericalSplit(split_index)
                children_list = self.split.apply_split(x, y, min_samples_leaf, random_state)
            elif FEATURE_TYPE[data_feature_types[split_index]] == "Nominal":
                self.split = NominalSplit(split_index)
                children_list = self.split.apply_split(x, y, min_samples_leaf, nominal_feature_values)
                # update the list of nominal features used
                # we can safely do this here even if the split will not be created because if we can not split on that
                # feature here, we will also not be able to split at any point later in that path
                new_nominal_splits.append(split_index)
            else:
                raise RuntimeError("Unexpected error in \"process_split_request\", error number: 1")

            # check, whether a split was successfully made and either try another split or continue with the current
            # split
            number_tries += 1
            # try again, if no children were made or the number of maximum tries (= number of features) is exceeded
            if len(children_list) != 0 or number_tries >= x.shape[1]:
                trying_splits = False

        # if the split did not return any children, return None
        if len(children_list) == 0:
            return None

        # create next Nodes
        for child in children_list:
            self.children.append(Node(child[1], len(self.classes), self.depth + 1, next_id))
            next_id += 1

        # return children and nominal_split list
        child_nominal_tuple = [(child, new_nominal_splits, child_x_y[0], child_x_y[1]) for child, child_x_y
                               in zip(self.children, children_list)]
        return child_nominal_tuple

    def predict_instances(self, new_x, nominal_feature_values, empty_leafs, verbose):
        """
        For every instance in new_x, return the classes and number of instances per class for the leaf node
        corresponding to that instance.
        :param new_x: instances (features) to get the predicted label for
        :param nominal_feature_values: list of all possible values for each nominal feature
        :param empty_leafs: what should be done when encountering empty leafs (see notes)
        :param verbose: if True, print additional information to the console
        :return: a numpy array with the respective leaf classes if empty_leafs != 2, a list-array structure otherwise
        """
        if verbose:
            print("Visiting node " + str(self.node_id))
            if empty_leafs == 2:
                print("Note the different data format because of the parameter empty_leafs == 2")

        # if this is a leaf, return y
        if len(self.children) == 0:
            return np.tile(self.classes, (new_x.shape[0], 1))

        # move in the direction corresponding to the split
        # to get the node, use self.children[number]
        next_nodes = [number for number in self.split.move_to_next_node(new_x, nominal_feature_values)]

        # if there is an "imaginary" child node (node number = len(self.children), does not exist), "None" is added
        # to the list where this None child will be treated like an empty node
        if len(self.children) not in next_nodes:
            children_plus = self.children
        else:
            children_plus = self.children + [None]

        # iterate over all children (iterating over instances would result in visiting children multiple times)
        children_results = []
        for number, child in enumerate(children_plus):
            # indices of instances which prediction path leads to that child node
            indices_for_child = np.where(np.array(next_nodes) == number)[0]
            # if no prediction paths lead to that child node, return an empty array
            if indices_for_child.shape[0] == 0:
                result = np.array([])
            else:
                # get the instances corresponding to the indices
                x_for_child = np.take(new_x, indices_for_child, 0)
                # if that child has zero instances and we do not want to use the empty_leafs approach which ignores
                # the prediction (in which case the zeros array can be returned anyway)
                if child is None or (not (np.any(child.classes)) and empty_leafs != 1):

                    # here is where empty_leafs comes into play

                    # return the parent node classes instead (current node)
                    if empty_leafs == 0:
                        result = np.tile(self.classes, (x_for_child.shape[0], 1))
                    # recursively go through siblings, return list with one result for each alternative path
                    elif empty_leafs == 2:
                        # for every sibling of next_node
                        sibling_res = []
                        for sibling in self.children:
                            if np.any(sibling.classes):
                                sibling_res.append(list(
                                    sibling.predict_instances(x_for_child, nominal_feature_values, empty_leafs,
                                                              verbose)))
                        # combine the sibling results into a useful data structure: a list, where each element
                        # represents the results for one instance, a list of arrays
                        result = []
                        for i in range(x_for_child.shape[0]):
                            temp = []
                            for s_res in sibling_res:
                                temp.append(s_res.pop(0))
                            result.append(temp)
                    # recursively go through siblings, return aggregated result of alternative paths
                    elif empty_leafs == 3:
                        # for every sibling of next_node
                        sibling_res = []
                        for sibling in self.children:
                            if np.any(sibling.classes):
                                sibling_res.append(sibling.predict_instances(x_for_child, nominal_feature_values,
                                                                             empty_leafs, verbose))
                        # sum those results as if they belonged to one node only
                        result = np.sum(sibling_res, axis=0)
                    # only possible if child is None (new nominal feature)
                    elif empty_leafs == 1:
                        # set result to 0 for every class for every instance
                        result = np.tile(np.zeros_like(self.classes), (x_for_child.shape[0], 1))
                    else:
                        raise ValueError("The \"empty_leafs\" parameter must be one of [0, 1, 2, 3]")
                else:
                    # child node has instances, so recursively predict on child node
                    result = child.predict_instances(x_for_child, nominal_feature_values, empty_leafs, verbose)
            # this list contains one element for each child node
            children_results.append(result)

        # different result structure (list instead of array) for empty_leafs == 2
        if empty_leafs == 2:
            result = []
        # otherwise, fixed size result array
        else:
            result = np.zeros((new_x.shape[0], self.classes.shape[0]), dtype=int)
        # stick the children results together, so that each prediction is assigned the corresponding instance
        for index, number in enumerate(next_nodes):
            element = children_results[number][0]
            children_results[number] = children_results[number][1:]
            if empty_leafs == 2:
                result.append(element)
            else:
                result[index] = element
        return result

    def predict_leafs(self, new_x, nominal_feature_values, verbose):
        """
        For every instance in new_x, return the leaf id for the prediction of that instance.
        For the case where the leaf does not exist (new nominal feature), the parent node is used.
        Empty leaf ids are returned.
        :param new_x: instances (features) to get the leafs for
        :param nominal_feature_values: list of all possible values for each nominal feature
        :param verbose: if True, print additional information to the console
        :return: a list of leafs
        """
        if verbose:
            print("Visiting node " + str(self.node_id))

        # if this is a leaf, return it
        if len(self.children) == 0:
            return [self.node_id] * new_x.shape[0]

        # move in the direction corresponding to the split
        # to get the node, use self.children[number]
        next_nodes = [number for number in self.split.move_to_next_node(new_x, nominal_feature_values)]

        # if there is an "imaginary" child node (node number = len(self.children), does not exist), "None" is added
        # to the list where this None child will be treated like an empty node
        if len(self.children) not in next_nodes:
            children_plus = self.children
        else:
            children_plus = self.children + [None]

        # iterate over all children (iterating over instances would result in visiting children multiple times)
        children_results = []
        for number, child in enumerate(children_plus):
            # indices of instances which prediction path leads to that child node
            indices_for_child = np.where(np.array(next_nodes) == number)[0]
            # if no prediction paths lead to that child node, return an empty array
            if indices_for_child.shape[0] == 0:
                result = []
            else:
                # get the instances corresponding to the indices
                x_for_child = np.take(new_x, indices_for_child, 0)
                if child is None:
                    result = [self.node_id] * x_for_child.shape[0]
                else:
                    result = child.predict_leafs(x_for_child, nominal_feature_values, verbose)
            # this list contains one element for each child node
            children_results.append(result)

        result = [None] * new_x.shape[0]
        # stick the children results together, so that each prediction is assigned the corresponding instance
        for index, number in enumerate(next_nodes):
            # get the leaf index
            element = children_results[number][0]
            # delete the leaf index, so that the leaf for the next index can be accessed with the line above
            children_results[number] = children_results[number][1:]
            # set the leaf index in the result list
            result[index] = element
        return result


class Tree:
    """
    This class represents a tree. This class keeps information about the tree structure and the instances used for
    building that tree.
    """

    def __init__(self, x, y, num_classes, data_feature_types):
        """
        Create a tree instance and declare a root Node.
        :param x: feature data
        :param y: data labels
        :param num_classes: number of classes of the entire dataset (node can have less classes)
        :param data_feature_types: types of each feature (nominal or numeric)
        """

        self.x = x
        self.y = y
        self.num_classes = num_classes

        # set data_feature_types and nominal_feature_values
        if data_feature_types is None:
            df_info = set_data_feature_information(self.x)
            self.data_feature_types, self.nominal_feature_values = df_info
        else:
            self.data_feature_types = data_feature_types
            self.nominal_feature_values = set_nominal_feature_values(self.x, data_feature_types)

        # each node has a unique id
        self.id_counter = 0

        self.root = None

    def build_tree(self, max_depth, min_samples_split, min_samples_leaf, random_state, verbose):
        """
        Given a set of train instance (x, y) and tree building criteria, fit this tree to the data. This will result in
        a random decision tree.
        :param max_depth: maximum depth of the tree (parameter for RDT)
        :param min_samples_split: minimum number of instances to try to split (parameter for RDT)
        :param min_samples_leaf: minimum number of instances in a leaf node (parameter for RDT, exceptions exist)
        :param random_state: random state, can be used for deterministic behavior
        :param verbose: if True, print additional information to the console
        """

        # always build a tree with x and y if x and y are valid (may violate
        # the max_depth or min_samples_leaf criteria)
        self.root = Node(self.y, self.num_classes, 0, self.id_counter)
        self.id_counter += 1

        # the work_list contains tuples of nodes, which need to be considered
        # for splitting
        # each tuple contains the respective node and a list, containing the
        # feature numbers of previous nominal splits (to avoid splitting a
        # nominal feature twice (or more often))
        work_list = [(self.root, [], self.x, self.y)]
        while len(work_list) > 0:
            # process current node
            next_tuples = work_list[0][0].process_split_request(work_list[0][2], work_list[0][3], max_depth,
                                                                min_samples_split, min_samples_leaf, random_state,
                                                                work_list[0][1], self.data_feature_types,
                                                                self.nominal_feature_values, self.id_counter, verbose)
            # add new node (children nodes) and increase id_counter
            if next_tuples is not None:
                work_list += next_tuples
                self.id_counter += len(next_tuples)
            # remove processed node
            work_list.pop(0)

        return

    def get_leafs(self):
        """
        Return a list of all leaf nodes of this tree.
        :return: list of all nodes of the tree which are leafs
        """
        leafs = []
        work_list = [self.root]
        # breath first
        while len(work_list) != 0:
            node = work_list.pop()
            if node.is_leaf():
                leafs.append(node)
            else:
                work_list.extend(node.children)
        return leafs

    def predict(self, new_x, empty_leafs, verbose):
        """
        For each instance in new_x, predict the classes using this tree.
        :param new_x: instances for which the children (ids) are returned
        :param empty_leafs: what should be done when encountering empty leafs (see notes)
        :param verbose: if True, print additional information to the console
        :return: a numpy array with the respective leaf classes if empty_leafs != 2, otherwise a list-array structure
        """
        return self.root.predict_instances(new_x, self.nominal_feature_values, empty_leafs, verbose)

    def predict_leafs(self, new_x, verbose):
        """
        For each instance in new_x, predict the leaf using this tree.
        :param new_x: instances for which the children (ids) are returned
        :param verbose: if True, print additional information to the console
        :return: a numpy array with the respective leafs
        """
        return self.root.predict_leafs(new_x, self.nominal_feature_values, verbose)

    def predict_class(self, new_x, empty_leafs, verbose):
        """
        For each instance in new_x, predict the classes using this tree (no method needs to be specified, just use the
        majority class of the leaf).
        :param new_x: instances for which the children (ids) are returned
        :param empty_leafs: what should be done when encountering empty leafs (see notes)
        :param verbose: if True, print additional information to the console
        :return: a numpy array with the respective prediction if empty_leafs != 2, a list-array structure otherwise
        """
        if empty_leafs == 2:
            print("To predict classes using only this tree, empty_leafs=3 is used instead of empty_leafs=2")
            empty_leafs = 3
        return np.argmax(self.root.predict_instances(new_x, self.nominal_feature_values, empty_leafs, verbose), axis=1)

    def predict_value(self, new_x, empty_leafs, mode, verbose):
        """
        Executes the given function. Calculates parameters for these functions if necessary.
        :param new_x: instances for which the children (ids) are returned
        :param empty_leafs: what should be done when encountering empty leafs (see notes)
        :param mode: which prediction mode / method / function to use for prediction
        :param verbose: if True, print additional information to the console
        :return: a numpy array with the respective scores if empty_leafs != 2, a list-array structure otherwise
        """
        classes = self.predict(new_x, empty_leafs, verbose)

        leafs_per_instance = None
        if empty_leafs == 2:
            classes, leafs_per_instance = prepare_empty_leafs_2(classes)

        if classes.shape[1] != 2:
            raise ValueError("This implementation can only work with two classes. The implementation should be updated "
                             "or a different dataset or the simple class prediction method should be used.")

        to_return = prediction_methods.method_switcher(classes, mode, np.sum(self.root.classes))

        if empty_leafs == 2:
            to_return = finish_empty_leafs_2(to_return, mode, new_x, leafs_per_instance)

        if len(to_return.shape) == 1:
            to_return = to_return.reshape((to_return.shape[0], 1))

        return to_return

    def print_tree(self):
        """
        Print a fairly simple description of the tree structure. In doing so, some information for every node is
        printed.
        """
        work_list = [self.root]
        while len(work_list) > 0:
            # print node information
            print_node(work_list[0])
            # add further nodes (if existing)
            if work_list[0].children is not None:
                work_list += work_list[0].children
            # remove already printed node from worklist
            work_list.pop(0)
        return


class RandomDecisionTree:
    """
    A class representing a random decision tree classifier.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 random_state=None):
        """
        Create a random decision tree. A random seed can be specified to create a random state instance which allows for
        deterministic behavior.
        :param max_depth: maximum allowed depth of the tree
        :param min_samples_split: minimum number of instances to try to split
        :param min_samples_leaf: minimum number of instances in a leaf node
        :param random_state: random state, can be used for deterministic behavior
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_converter = None
        # set random_state
        if random_state is None:
            self.random_state = np.random.mtrand._rand
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            raise TypeError("random_state must be a numpy random state, an int or None")

    def fit(self, x, y, data_feature_types=None, verbose=False):
        """
        Fit the tree structure.
        :param x: feature array of the dataset
        :param y: labels of the dataset
        :param data_feature_types: types of each feature (nominal or numeric)
        :param verbose: if True, print additional information to the console
        """
        # check, that the data has a valid structure and valid class types
        checked_x, checked_y, num_classes, self.class_converter = check_data(x, y, suppress_info=not verbose)

        self.tree = Tree(checked_x, checked_y, num_classes, data_feature_types)
        self.tree.build_tree(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.random_state, verbose)

    def predict(self, x, empty_leafs=3, mode="class", verbose=False, y=None):
        """
        Predict the classes corresponding to the set of instances x. The mode corresponds to the type of results wanted
        (e.g. "instances" returns the classes of the instances themselves in an array). Include y, if you know the true
        labels (which is y) and want to apply the corresponding preprocessing (label transformation). In that case, the
        returned object will be a tuple with the true classes (transformed labels) as the FIRST tuple element, and the
        prediction as second (making it easy to apply to an accuracy function, for example).
        :param x: feature array of the dataset for which to make predictions
        :param empty_leafs: what should be done when encountering empty leafs (see notes)
        :param mode: which prediction mode / method / function to use for prediction
        :param verbose: if True, print additional information to the console
        :param y: true classes, if not None, the "checked" version of y will be returned (see check_data)
        :return: the prediction if y is None, else a tuple of the "checked" y and the prediction
        """
        x = check_data(x, suppress_info=not verbose)
        if mode == "instances":  # leaf instances
            prediction = self.tree.predict(x, empty_leafs, verbose)
        elif mode == "leafs":  # leaf ids
            prediction = self.tree.predict_leafs(x, verbose)
        elif mode == "class":  # class prediction
            prediction = self.tree.predict_class(x, empty_leafs, verbose)
        else:  # applying the prediction function (mode) on the leaf instances
            prediction = self.tree.predict_value(x, empty_leafs, mode, verbose)
        if y is None:
            return prediction
        else:
            _, checked_y, _, _ = check_data(x, y, not verbose)
            if verbose:
                print("Returning a tuple with the true labels as first tuple element and the prediction as second.")
            return checked_y, prediction

# Notes:
# Nominal splits will only be applied once (or not at all) per path. This means that it is also possible to split when
# all instances have the same feature value for that split (if min_samples_leaf > 0). However, this might still matter
# if a new instance (test instance) has a different value, in which case it is directed into an empty leaf.
#
# A design decision was made which applies the min_samples_leaf criterion to nominal splits in a way, such that the two
# largest children must fulfill the criterion. In cases where we have only one distinct value for a nominal feature,
# this check is only applied (and if applied, fails every time) if min_samples_leaf is not 0 (with 0 meaning, we do not
# restrict the leaf size in any way).
#
# The empty_parameter of the prediction corresponds to which method is used when reaching empty leaf nodes. Options:
# 0: Use the parent node instead.
# 1: Ignore that prediction (basically still use that leaf, 0 for every class).
# 2: Recursively iterate the siblings, one prediction for each resulting leaf.
# 3: Recursively iterate the siblings, aggregate leafs for one prediction.
#
# Also note, that this random tree implementation does not consider the classes when splitting. Therefore, it is
# entirely possible (and will happen quite often) that a node is split even if it only has instances of one class.
#
# Lastly, keep in mind that setting min_samples_leaf to 0 might result in infinite loops, for example if two identical
# instances exist in the dataset.
