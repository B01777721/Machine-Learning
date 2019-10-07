import numpy as np
import random
import copy
import math

def train_error(dataset):
    '''
        TODO:
        Calculate the train error of the subdataset and return it.
        For a dataset with two classes:
        C(p) = min{p, 1-p}
    '''
    m = len(dataset) # number of examples in the dataset
    if m == 0:
        return 0
    else:
        count = 0
        for i in range(0,m):
            if dataset[i][0] == 1:
                count += 1
        p = count / m
        return min(p,1-p)

def entropy(dataset):
    '''
        TODO:
        Calculate the entropy of the subdataset and return it.
        This function is used to calculate the entropy for a dataset with 2 classes.
        Mathematically, this function return:
        C(p) = -p*log(p) - (1-p)log(1-p)
    '''
    m = len(dataset) # number of examples in the dataset
    #print(m)
    if m == 0:
        return 0
    else:    
        count = 0
        for i in range(0,m):
            if dataset[i][0] == 1:
                count += 1
        p = count / m
        #print("entropy: " + str(p))
        if p == 0 or p == 1:
            return 0
        else:
            return -p*math.log2(p) - (1-p)*math.log2(1-p)

def gini_index(dataset):
    '''
        TODO:
        Calculate the gini index of the subdataset and return it.
        For dataset with 2 classes:
        C(p) = 2*p*(1-p)
    '''
    m = len(dataset) # number of examples in the dataset
    if m == 0:
        return 0
    else:
        count = 0
        for i in range(0,m):
            if dataset[i][0] == 1:
                count += 1
        p = count / m
        return 2*p*(1-p)



class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1, info={}):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = info


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))
        #print(indices)
        self._split_recurs(self.root, data, indices)

        # Pruning
        if not (validation_data is None):
            self._prune_recurs(self.root, validation_data)

    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)

    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)

    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)

    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        '''
        if node.isleaf:
            return

        self._prune_recurs(node.left,validation_data)
        self._prune_recurs(node.right,validation_data)

        if node.left.isleaf and node.right.isleaf:
            original_loss = self.loss(validation_data)

            node.isleaf = True
            node.label = 1
            loss1 = self.loss(validation_data)

            node.label = 0
            loss0 = self.loss(validation_data)

            node.isleaf = False
            node.label = -1
            tempNode = node
            node = node.left
            loss_left = self.loss(validation_data)

            node = tempNode
            node = node.right
            loss_right = self.loss(validation_data)

            node = tempNode

            list1 = [original_loss,loss0,loss1,loss_left,loss_right]
            arr1 = np.array(list1)
            indx = np.argmin(arr1)

            if indx == 0:
                node.isleaf = False
                node.label = -1
            elif indx == 1:
                node.isleaf = True
                node.label = 0
            elif indx == 2:
                node.isleaf = True
                node.label = 1
            elif indx == 3:
                node.isleaf = False
                node = node.left
                node.label = -1
            else:
                node.isleaf = False
                node = node.right
                node.label = -1

            '''if original_loss < loss1 and original_loss < loss0:
                node.isleaf = False
                node.label = -1

            if loss1 <= loss0:
                node.label = 1'''
       


    def _is_terminal(self, node, data, indices):
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the nodex exceede the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf.
            - A label, indicating the label of the leaf (-1 if False)
        '''
        if len(data) == 0: # if dataset is empty
            return (True,random.randint(0,1))
        if all(d[0] == data[0][0] for d in data): #if all instances belong to the same class
            return (True,data[0][0]) # return True,node.label
        if len(indices) == 0 or node.depth > self.max_depth:
            frac = self._majority_vote(data)
            if frac <= 0.5:
                c = 0
            else:
                c = 1
            return (True,c)
        else:
            return (False,-1)

    def _majority_vote(self,data):
        m = len(data)
        count = 0
        for d in data:
            if d[0] == 1:
                count += 1
        frac = count / m
        return frac

    def _split_recurs(self, node, rows, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.
        First use _is_terminal() to check if the node needs to be splitted.
        Then select the column that has the maximum infomation gain to split on.
        Also store the label predicted for this node.
        Then split the data based on whether satisfying the selected column.
        The node should not store data, but the data is recursively passed to the children.
        '''
        (terminate,labeling) = self._is_terminal(node,rows,indices)
        if terminate == True:
            node.isleaf = terminate
            node.label = labeling
        else:
            node.isleaf = terminate
            '''if this node is not terminal we have to split this node'''
            # check bestGain, bestFeature
            bestFeature = indices[0]
            bestGain = self._calc_gain(rows,bestFeature,self.gain_function)

            for index in indices:
                result = self._calc_gain(rows,index,self.gain_function)
                if result >= bestGain:
                    bestGain = result
                    bestFeature = index

            fraction = self._majority_vote(rows)
            if fraction <= 0.5:
                node.label = 0
            else:
                node.label = 1

            node.index_split_on = bestFeature #??????? update here?

            ''' bestFeature = the index that maximizes gain function'''
            ''' split the rows into two data using index-column'''
            trueRow = []
            falseRow = []
            for r in range(0,len(rows)):
                if rows[r][bestFeature]:
                    tlist = rows[r] 
                    trueRow.append(tlist)
                else:
                    flist = rows[r]
                    falseRow.append(flist)

            new_indices = [i for i in indices if i != bestFeature]

            'call _split_recurs recursively for each left/right node'
            node.left = Node(depth=node.depth+1)
            self._split_recurs(node.left,trueRow,new_indices)
            node.right = Node(depth=node.depth+1) 
            self._split_recurs(node.right,falseRow,new_indices)




    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - (P[x_i=True] * C(P[y=1|x_i=True]) + P[x_i=False]C(P[y=1|x_i=False)])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        # compute C(P[y=1])
        CP = gain_function(data)
        m = len(data) # number of instances
        if m == 0:
            return 0
        else:
            count = 0
            trueDataSet = []
            falseDataSet = []
            for i in range(0,m):
                if data[i][split_index] == True:
                    count += 1
                    trueDataSet.append(data[i])
                else:
                    falseDataSet.append(data[i])

            ptrue = count / m 
            pfalse = 1 - ptrue

        # compute CPtrue and CPfalse
        CPtrue = gain_function(trueDataSet)
        CPFalse = gain_function(falseDataSet)
        Gain = abs(CP - ptrue*CPtrue - pfalse*CPFalse)
        return Gain
    


    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        temp = []
        output = []
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = %d; cost = %f; sample size = %d' % (node.index_split_on, node.info['cost'], node.info['data_size'])
            left = indent + 'T -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + 'F -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')




    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)



    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
