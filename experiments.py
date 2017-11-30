import numpy as np
import scipy as sp
from sklearn.linear_model import SGDClassifier
import util
import matplotlib.pyplot as plt
import active


def active_learn(mat, gold_rel, turk_rel, n = 200, t = 100, algo = 'random', sep = 2000):
    """
    Do active learning experiments
    mat = data
    gold_rel = gold data
    turk_rel = turk data
    
    n = # sample to take
    t = # experiments
    
    mat[:sep] is for training, mat[sep:] is for testing
    """
    list_ap = []
    for i in range(t):
        print i
        (clf, adata) = active.run(mat[:sep,:], turk_rel[:sep], n, algo) # train on turk data
        n1 = np.sum(adata.taken_rel)
        ap = util.eval_clf(gold_rel[sep:], clf, mat[sep:,:]) # evaluate on gold
        print n1, ap
        list_ap.append(ap)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 20.0])
    plt.xlabel('AUC')
    plt.ylabel('Freq')    
    plt.title(algo + ' Sampling')
    plt.hist(list_ap, bins = np.arange(0,1,0.05))
    return list_ap
    
    
def active_learn_with_crowd(mat, turk_rel, n = 200, algo = 'random', sep = 2000):
    (clf, adata) = active.run(mat[:sep,:], turk_rel[:sep], n, algo) # train on turk data
    return (clf, adata)
    
    
def active_learn_with_expert(adata, turk_uncer, gold, n = 20, algo = 'random', sep = 2000):
    """
    Do active learning by querying expert, 
    given crowd labels
    """
    pass
    
def new(mat, gold_rel, turk_rel, turk_uncer, t = 100, n = 500, algo = 'random', sep = 2500):
    res = []
    for i in range(t):
        # active learn by querying turk
        (clf, adata) = active.run(mat[:sep,:], turk_rel[:sep], n, algo)
        clf_ap = util.eval_clf(gold_rel[sep:], clf, mat[sep:])
        print 'first', clf_ap
        # active learn on turk uncertainty, querying expert
        (adata_exp, new_clf, new_rel) = active.run_al_with_expert(adata, turk_uncer[:sep], gold_rel[:sep], 50, 'uncertainty')
        new_clf_ap = util.eval_clf(gold_rel[sep:], new_clf, mat[sep:,:])
        print 'after expert', new_clf_ap
        res.append(new_clf_ap - clf_ap)
        print '------------------------------------------------------'
    return res
    
    
    
    
