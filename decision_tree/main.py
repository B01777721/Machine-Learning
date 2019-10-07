import random
import numpy as np
import matplotlib.pyplot as plt

from get_data import get_data
from decision_tree import DecisionTree, train_error, entropy, gini_index


def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)

    # TODO: Print 12 loss values associated with the dataset.
    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)

    tree1 = DecisionTree(train_data,gain_function=train_error)
    print(str(filename) + " Train_error " + "Training loss (No Prune) = " + str(tree1.loss(train_data)))
    print(str(filename) + " Train_error " + "Testing loss  (No Prune) = " + str(tree1.loss(test_data)))
    tree1._prune_recurs(tree1.root,validation_data)
    print(str(filename) + " Train_error " + "Training loss (Pruning)  = " + str(tree1.loss(train_data)))
    print(str(filename) + " Train_error " + "Test loss     (Pruning)  = " + str(tree1.loss(test_data)))   

    tree2 = DecisionTree(train_data,gain_function=entropy)
    print(str(filename) + " Entropy     " + "Training loss (No Prune) = " + str(tree2.loss(train_data)))
    print(str(filename) + " Entropy     " + "Testing loss  (No Prune) = " + str(tree2.loss(test_data)))
    tree2._prune_recurs(tree2.root,validation_data)
    print(str(filename) + " Entropy     " + "Training loss (Pruning) = " + str(tree2.loss(train_data)))
    print(str(filename) + " Entropy     " + "Test loss     (Pruning) = " + str(tree2.loss(test_data)))   

    tree3 = DecisionTree(train_data,gain_function=gini_index)
    print(str(filename) + " Gini Index  " + "Training loss (No Prune) = " + str(tree3.loss(train_data)))
    print(str(filename) + " Gini Index  " + "Testing loss  (No Prune) = " + str(tree3.loss(test_data)))
    tree3._prune_recurs(tree3.root,validation_data)
    print(str(filename) + " Gini Index  " + "Training loss (Pruning)  = " + str(tree3.loss(train_data)))
    print(str(filename) + " Gini Index  " + "Test loss     (Pruning)  = " + str(tree3.loss(test_data)))   

'''
    x = []
    noprune_loss = []
    prune_loss = []
    
    for i in range(1,16):
        x.append(i)
        plotTree = DecisionTree(train_data,gain_function = entropy,max_depth=i)
        loss = plotTree.loss(train_data)
        noprune_loss.append(loss)

        plotTreePrune = DecisionTree(train_data,validation_data=validation_data,gain_function=entropy,max_depth=i)
        pruneLoss = plotTreePrune.loss(train_data)
        prune_loss.append(pruneLoss)

    plt.plot(x,noprune_loss,'.b',label='no Prune')
    plt.plot(x,prune_loss,'.r',label='Prune')
    plt.legend()
    plt.title('Training Data Loss for Entropy')
    plt.xlabel('Max_depth of tree')
    plt.ylabel('Loss')
    plt.show() '''
    
    
        

    
    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!

def main():
    ########### PLEASE DO NOT CHANGE THESE LINES OF CODE! ###################
    random.seed(1)
    np.random.seed(1)
    #########################################################################

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

    

main()
