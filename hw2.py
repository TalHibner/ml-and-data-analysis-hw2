import numpy as np
import matplotlib.pyplot as plt

'''
ID_1: 342681004
ID_2: 026548446
'''


### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    if len(data) > 0:
        labels = data[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        gini = 1.0 - np.sum(probabilities ** 2)
    
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    if len(data) > 0:
        labels = data[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
    

    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        if len(self.data) > 0:
            labels, counts = np.unique(self.data[:, -1], return_counts=True)
            pred = labels[np.argmax(counts)]
        '''target = self.data[:,-1]
        count = {}
        for i in range(len(target)):
            if target[i] in count:
                count[target[i]] += 1
            else:
                count[target[i]] = 1
        pred = max(count, key=count.get)'''
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        if self.feature != -1:
            original_gain_ratio = self.gain_ratio
            self.gain_ratio = False
            goodness, _ = self.goodness_of_split(self.feature)
            self.gain_ratio = original_gain_ratio
            
            self.feature_importance = (len(self.data) / n_total_sample) * goodness
        else:
            self.feature_importance = 0.0
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        if len(self.data) == 0:
            return 0.0, {}

        feature_values = self.data[:, feature]
        unique_values, counts = np.unique(feature_values, return_counts=True)
        total_samples = len(self.data)
        
        impurity_func = calc_entropy if self.gain_ratio else self.impurity_func
        impurity_S = impurity_func(self.data)
        
        sum_impurity_sv = 0.0
        split_information = 0.0
        
        for val, count in zip(unique_values, counts):
            sv = self.data[feature_values == val]
            groups[val] = sv
            prob = count / total_samples
            sum_impurity_sv += prob * impurity_func(sv)
            if self.gain_ratio:
                split_information -= prob * np.log2(prob)
                
        goodness = impurity_S - sum_impurity_sv
        
        if self.gain_ratio:
            goodness = goodness / split_information if split_information != 0 else 0.0
        
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Check if we reached the max depth or not yet
        if self.depth >= self.max_depth:
            self.terminal = True
            return
        
        # Check if the node is already pure or not
        if self.impurity_func(self.data) == 0:
            self.terminal = True
            return
        
        # Check if we have a feature that give us a useful split, otherwise we already reached the terminal node
        best_goodness = -float('inf')
        best_feature = -1
        best_groups = None  
        n_total_sample = len(self.data)
        for feature in range(self.data.shape[1] - 1):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups  

        if best_goodness <= 0:
            self.terminal = True
            return
        
        # Check for pruning
        if self.chi<1:
            chi_square = 0.0
            classes, counts = np.unique(self.data[:, -1], return_counts=True)
            parent_probs = counts / len(self.data)
            class_prob = dict(zip(classes, parent_probs))
            for group in best_groups.values():
                n_f = len(group)
                g_classes, g_count = np.unique(group[:,-1], return_counts=True)
                observed = dict(zip(g_classes,g_count))

                for clss in classes:
                    expected = n_f * class_prob[clss]
                    actual = observed.get(clss, 0)
                    if expected>0:
                        chi_square += ((actual - expected)**2) / expected

            deg_of_freedom = (len(best_groups)-1) * (len(classes) - 1)
            crit_value = chi_table.get(deg_of_freedom, {}).get(self.chi,float('inf'))
            
            if chi_square < crit_value:
                self.terminal = True
                return
            
        self.feature = best_feature
        for v,g in best_groups.items():
            child = DecisionNode(
            data=g,
            impurity_func=self.impurity_func,
            feature=-1,
            depth=self.depth + 1,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
            )
            self.add_child(child,v)
            child.split()
            
            

                    
class DecisionTree:
    def __init__(self, data, impurity_func, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio,
            depth=0,
            feature=-1
        )
        self.root.split()

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: a row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        r = self.root
        while not r.terminal:
            spliting_feature = r.feature
            instance_value = instance[spliting_feature]
            children_found = False
            for child_idx in range(len(r.children_values)):
                if r.children_values[child_idx] == instance_value:
                    r = r.children[child_idx]
                    children_found = True
                    break
            if not children_found:
                break
        node = r
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        count = 0
        for row in dataset:
            pred = self.predict(row)
            if pred == row[-1]:
                count += 1
        accuracy = (count / len(dataset))*100
        return accuracy
        
    def depth(self):
        if self.root is None:
            return 0
        
        def calc_depth_node(node):
            if node.terminal or len(node.children) == 0:
                return node.depth
            return max(calc_depth_node(child) for child in node.children)
        
        return calc_depth_node(self.root)

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(data=X_train, 
                            impurity_func=calc_entropy, 
                            chi=1, 
                            max_depth=max_depth, 
                            gain_ratio=True)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))

    return training, validation


def chi_pruning(X_train, X_validation):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    chiS = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for c in chiS:
        tree = DecisionTree(data=X_train, 
                            impurity_func=calc_entropy, 
                            chi=c, 
                            max_depth=1000, 
                            gain_ratio=True)
        tree.build_tree()
        #depth.append(count_nodes(tree.root))
        depth.append(tree.depth())
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_validation))
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    n_nodes = 1
    if node.terminal:
        return n_nodes
    else:
        for child in node.children:
            n_nodes += count_nodes(child)
    return n_nodes






