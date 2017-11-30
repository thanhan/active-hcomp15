import numpy as np
import scipy as sp
from sklearn.linear_model import SGDClassifier
import util
import matplotlib.pyplot as plt

from scipy.stats import binom


import random

import sklearn

import copy

from sklearn.utils import resample

import logging

# loss if fail to recall an article (precision loss = 1)
#RECALL_LOSS = 10
PRIOR_COUNT = [[100,1],[1,100]]
#PRIOR_COUNT = [[1,1],[1,1]]
#PRIOR_COUNT = [[10,1],[1,10]]
WINDOW_SIZE = 1000000

#PRIOR_COUNT = [[100,100],[100,100]] # crowd bad 



class active_data:
    def __init__(self, mat, rel, gold_rel = None, crowd_label_count = None, money = (12500, 5, 100), clf = None):
        """
        mat = data
        rel = relevance label
        """
        (m, n) = mat.shape # m items, n features
        
        self.mat = mat
        self.rel = np.asarray(rel)
        self.items = range(m)
        self.gold_rel = gold_rel
        
        self.spent_crowd = 0
        self.budget = money[0]
        self.crowd_cost = money[1]
        self.expert_cost = money[2]
        
        self.taken_mat = None
        self.taken_rel = None
        self.taken_items = []
        self.expert_fixed = []
        self.label_status = []
        self.sample_weight = []
        self.eval_rel = None # taken rel used for evaluation
        
        #count[i][j] = crowd label i for true label j
        self.count = PRIOR_COUNT
        
        self.taken_crowd_labels = []
        
        if crowd_label_count != None:
            self.create_individual_crowd_label(crowd_label_count)
        
        #print self.count
        
    def create_individual_crowd_label(self, turk_uncer):
        """
        create crowd labels from the label count (turk_uncer)
        """
        n = len(turk_uncer)
        #print turk_uncer
        self.crowd_labels = []
        for i in range(n):
            c0 = turk_uncer[i][0]
            c1 = turk_uncer[i][1]
            self.crowd_labels.append([0] * c0 + [1] * c1)
            np.random.shuffle(self.crowd_labels[i])
        
    def take(self, i, new_status = 1):
        """
        Item i is requested
        move it from data to taken data
        """
        res = self.rel[i]
        
        # add item i to taken data
        self.taken_items.append(self.items[i])
        self.expert_fixed.append(False)
        self.label_status.append(new_status)
        self.taken_crowd_labels.append(self.crowd_labels[i])
        
        if self.taken_mat == None:
            self.taken_mat = self.mat[i,:]
            self.taken_rel = np.asarray(self.rel[i:i+1])
        else:
            self.taken_mat = sp.sparse.vstack([self.taken_mat, self.mat[i,:]], format='csr')
            self.taken_rel = np.append(self.taken_rel, self.rel[i])
            
        # del item i from data
        del(self.items[i])
        if len(self.rel) == 1:
            self.mat = np.array([])
        elif i == len(self.rel) - 1:            # take the last
            self.mat = self.mat[:i,:]
        elif i == 0:                          # take the first
            self.mat = self.mat[1:,:]
        else:
            self.mat = sp.sparse.vstack([self.mat[:i,:], self.mat[i+1:,:]], format='csr')
        self.rel = np.delete(self.rel, i, 0)
        del self.crowd_labels[i]  
        
        return res
        
    def fix(self, i):
        """
        i = index of taken items
        fix the (crowd) label of taken_rel[i] using expert label
        """
        #if self.num_fix <= 0:
        #    return
        #self.num_fix -= 1
        
        self.expert_fixed[i] = True
        true_label = self.gold_rel[ self.taken_items[i] ]
        #self.count[ int(self.taken_rel[i]) ] [int(true_label)] += 1
        for l in self.taken_crowd_labels[i]:
            self.count[l] [int(true_label)] += 1
        self.taken_rel[i] = true_label
        #print self.count[0][0], self.count[0][1], self.count[1][0], self.count[1][1]
        
    def query_crowd_all(self, i, w = 1.0):
        """
        query all crowd labels for item i
        """
        N = len(self.crowd_labels[i])# number of crowd L available
        if self.budget < N*self.crowd_cost: return -1
        self.budget -= N*self.crowd_cost
        self.spent_crowd += N*self.crowd_cost
        w = w * 1.0 / 100
        if w > 10: w = 10
        self.sample_weight.append(w)
        return self.take(i, new_status = N)
            
    def query_expert_direct(self, i, w = 1):
        if self.budget < self.expert_cost: return -1
        self.budget -= self.expert_cost
        self.take(i)
        self.fix( len(self.taken_items) - 1)
        self.sample_weight.append(w)
        return 0
        
    def query_expert_fix(self, i):
        if self.budget < self.expert_cost: return -1
        self.budget -= self.expert_cost
        self.fix(i)
        return 0
        
    def query_crowd_single(self, i, w = 1.0):
        if self.budget < self.crowd_cost: return -1
        self.budget -= self.crowd_cost
        self.spent_crowd += self.crowd_cost
        self.rel[i] = self.crowd_labels[i][0]
        w = w * 1.0 / 100
        if w > 10: w = 10
        self.sample_weight.append(w)
        return self.take(i)
        
    def majority_vote(self, a):
        if sum(a) > (len(a) - sum(a)):
            return 1
        elif sum(a) < (len(a) - sum(a)):
            return 0
        return np.random.randint(2)
        
    def query_crowd_again(self, j):
        if self.budget < self.crowd_cost: return -1
        self.budget -= self.crowd_cost
        self.spent_crowd += self.crowd_cost
    
        #i = self.taken_items[j] # original index
        #self.crowd_labels[i]
        self.label_status[j] += 1
        num = self.label_status[j]
        self.taken_rel[j] = self.majority_vote(self.taken_crowd_labels[j][0:num])
        
    def taken_both_classes(self):
        if self.taken_rel == None:
            return False
        s = np.sum(self.taken_rel)
        n = len(self.taken_items)
        if s < 0.000001 or s > n-0.000001:
            return False
            
        return True
        
    def evaluate(self, clf, r):
        """
        given predicted class for unlabelled data
        """
        if self.mat.shape[0] > 0:
            predicted = clf.predict(self.mat)
        # total risk
        risk = 0
        # taken data:

        for i in range(len(self.taken_rel)):
            if self.eval_rel == None:
                y = int(self.taken_rel[i])
            else:
                y = int(self.eval_rel[i])
            x = int(self.gold_rel[ self.taken_items[i] ])
            risk += r[x][y]
            
        crowd_loss = risk
        #print "labeled data risk", risk
        # unlabelled data
        for i in range(len(self.rel)):
            x = int(self.gold_rel[ self.items[i] ])
            y = int(predicted[i])
            risk += r[x][y]
        unlab_loss = risk - crowd_loss
        print (unlab_loss, crowd_loss)
        logging.info("{0} {1}".format(unlab_loss, crowd_loss))
        #print "loss", r
        return risk
        
    def cheat_proba(self):
        pp = []
        for i in range(len(self.rel)):
            x = int(self.gold_rel[ self.items[i] ])
            if x == 1:
                pp.append([0.1, 0.9])
            else:
                pp.append([0.9, 0.1])
        return pp
        
##################################################################################################
# Active Learning on Crowd (level 1)

def random_sampling(mat):
    """
    Pick next examples at random
    """
    (m, n) = mat.shape
    return np.random.randint(m)
    
    
def uncertainty_sampling(adata, clf, i):
    """
    Pick most uncertain example
    """ 
    # Use random sampling at first
    if i <=20 or np.sum(adata.taken_rel) < 0.1: # no class 1 example
        return random_sampling(adata.mat)
        
    clf.fit(adata.taken_mat, adata.taken_rel)
    pp = clf.predict_proba(adata.mat)
    #print 'mean predict_proba in un sam = ', np.mean(pp[:,0])
    #print pp.shape
    #print np.histogram(pp[:,0], np.arange(-0.1,1.1,0.1))[0]
    uncertain = np.abs(pp[:,0] - 0.5)
    #unc_mean = np.mean(pp[:,0])
    #unc = np.abs(pp[:,0] - unc_mean)
    
    res = np.argmin(uncertain)
    #print len(uncertain), res
    #return random_sampling(adata.mat)
    return res
    
    
def output_entropy(pp):
    n, m = pp.shape
    h = 0
    for i in range(n):
        h += sp.stats.entropy(pp[i,:])
    return h
    
    
def min_exp_err(adata, clf, i):
    """
    Estimate expected error for each doc
    (if we use the classifier)
    select the max
    """
    # Use random sampling at first
    if i <=20 or np.sum(adata.taken_rel) < 0.1: # no class 1 example
        return random_sampling(adata.mat)
        
        
    m, n = adata.mat.shape
    score = np.zeros(m)
    
    clf.fit(adata.taken_mat, adata.taken_rel)
    pp = clf.predict_proba(adata.mat)
    for i in range(adata.mat):
        # assume i is of class 0
        new_mat = sp.sparse.vstack([adata.taken_mat, adata.mat[i,:]])
        new_rel = np.append(adata.taken_rel, 0.0)
        clf.fit(new_mat, new_rel)
        pp0 = clf.predict_proba(adata.mat)
        
        new_mat = sp.sparse.vstack([adata.taken_mat, adata.mat[i,:]])
        new_rel = np.append(adata.taken_rel, 1.0)
        clf.fit(new_mat, new_rel)
        pp1 = clf.predict_proba(adata.mat)
        
        score[i] = pp[i,0] * output_entropy(pp0) + pp[i,1] * output_entropy(pp1)
        
    return np.argmax(score)
    
    
adata = None
    
def run(mat, rel, n = 200, algo = 'random'):
    """
    run active learning 
    """
    clf = SGDClassifier(loss="log", penalty="l1")

    global adata
    adata = active_data(mat, rel)
    
    for i in range(n):
        if algo == 'random':
            j = random_sampling(adata.mat)
        elif algo == 'uncertainty':
            j = uncertainty_sampling(adata, clf, i)
            #print adata.mat.shape, adata.rel.shape, j
        elif algo == 'minexperr':
            j = min_exp_err(adata, clf, i)
        else:
            print 'no ', algo
            
        adata.take(j)
        
    #adata.taken_rel.reshape((n, ))
    #print 'before fit', adata.taken_mat.shape, adata.taken_rel.shape
    clf.fit(adata.taken_mat, adata.taken_rel)
    
    return (clf, adata)
    
################################################################################################
# Active Learning with Expert (level 2)

def random_sampling_expert(adata, uncer_label):
    (m, n) = adata.taken_mat.shape
    return np.random.randint(m) 
    
    
def uncer_sampling_expert(mat, uncer_label):
    """
    Uncertainty sampling given
     - mat = data
     - uncer_label = (#no, #yes) from crowd
    ask expert the doc that crowd is uncertain
    """
    uncer_prob = uncer_label[:,0] *1.0 / (uncer_label[:,0] + uncer_label[:,1])
    uncertain = np.abs(uncer_prob - 0.5)
    
    res = np.argmin(uncertain)
    return res
    
def run_al_with_expert(adata, turk_uncer, gold, n = 20, algo = 'random'):
    """
    After AL with crowd
    Run AL with expert
    
    """
    turk_uncer = np.asarray(turk_uncer)
    uncer_label = turk_uncer[adata.taken_items, :]
    gold_label  = gold[adata.taken_items]
    adata_exp = active_data(adata.taken_mat.tocsr(), gold_label)    
    
    new_rel = adata.taken_rel.copy()
    
    for i in range(n):
        if algo == 'random':
            j = random_sampling(adata_exp.mat)
        elif algo == 'uncertainty':
            j = uncer_sampling_expert(adata_exp.mat, uncer_label)
        else:
            print 'no ', algo
            
        adata_exp.take(j)
        new_rel[j] = gold_label[j]
        
    
    clf = SGDClassifier(loss="log", penalty="l1")
    
    clf.fit(adata.taken_mat.tocsr(), new_rel)
    return (adata_exp, clf, new_rel)
    
################################################################################################

def query_expert(adata, turk_uncer, algo):
    """
    Querying expert to fix label
    n = number of items taken so far
    algo = 
        - last10: random one from last 10
        - uncer: query any doc that crowd not certain
    """
    n = len(adata.taken_items)
    if algo == 'last10':    
        if n % 10 != 0: return
        x = np.random.randint(10)
        adata.fix(n - 1 - x)
    elif algo == 'uncer':
        i = adata.taken_items[n - 1] # index of last item
        if turk_uncer[i] == (0,5) or turk_uncer[i] == (5,0): return
        adata.fix(n-1)
    
    
    
def run_al_combined(mat, turk_rel, gold_rel, turk_uncer, n = 500, qe_algo = 'last10', qe_num = 50):
    """
    Active Learning on crowd
    Query expert as needed
    """
    clf = SGDClassifier(loss="log", penalty="l1")
    global adata
    adata = active_data(mat, turk_rel, gold_rel, qe_num)
    
    for i in range(n):
        j = uncertainty_sampling(adata, clf, i)
        adata.take(j)
        query_expert(adata, turk_uncer, qe_algo)
        
                    
    clf.fit(adata.taken_mat, adata.taken_rel)
    
    return (clf, adata)


################################################################################################
def experiment(mat, gold_rel, turk_rel, turk_uncer, n = 500, t = 100, sep = 2500, train_on_turk = True, qe_algo = 'last5', tt = ''):
    list_ap = []
    for i in range(t):
        print i
        if train_on_turk:
            # train on turk data
            (clf, adata) = run_al_combined(mat[:sep,:], turk_rel[:sep], gold_rel[:sep], turk_uncer[:sep], n, qe_algo) 
            
        else:
            # train on gold data
            (clf, adata) = run(mat[:sep,:], gold_rel[:sep], n, algo) 
            
        n1 = np.sum(adata.taken_rel)
        ap = util.eval_clf(gold_rel[sep:], clf, mat[sep:,:]) # evaluate on gold
        print n1, ap
        list_ap.append(ap)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 20.0])
    plt.xlabel('AUC')
    plt.ylabel('Freq')    
    plt.title(tt)
    plt.hist(list_ap, bins = np.arange(0,1,0.05))
    return list_ap
    
#################################################################################################
#Cost

def al_just_crowd(adata, clf, thresh = 3):
    n = len(adata.taken_items)        # examples taken
    m = adata.mat.shape[0]          # examples available
    if m < 1: return -1
    if n < thresh or not adata.taken_both_classes():
        i = random.randint(0, m-1)
        return adata.query_crowd_all(i)
        
    # uncertainty sampling
    clf.fit(adata.taken_mat, adata.taken_rel)
    pp = clf.predict_proba(adata.mat)
    uncertain = np.abs(pp[:,0] - 0.5)
    
    i = np.argmin(uncertain)
    return adata.query_crowd_all(i)
    
def al_just_expert(adata, clf, thresh_random = 3):
    n = len(adata.taken_items)        # examples taken
    m = adata.mat.shape[0]          # examples available
    if m < 1: return -1
    if n < thresh_random or not adata.taken_both_classes():
        i = random.randint(0, m-1)
        return adata.query_expert_direct(i)
        
    # uncertainty sampling
    
    # undersample:
    #mat, rel = undersam(adata.taken_mat.tocsr(), adata.taken_rel)
    #clf.fit(mat, rel)
    
    clf.fit(adata.taken_mat, adata.taken_rel)
    pp = clf.predict_proba(adata.mat)
    uncertain = np.abs(pp[:,0] - 0.5)
    
    i = np.argmin(uncertain)
    j = np.argmin(pp[:,0])
    #print pp[i,0]
    return adata.query_expert_direct(i)
    
    
    
    
def al_crowd_if_dis_expert(adata, clf, turk_uncer, min_items = 1000):
    res = al_just_crowd(adata, clf)
    if res == -1: return res
    
    #if (adata.budget - adata.expert_cost) *1.0 / adata.crowd_cost + len(adata.taken_items) < min_items:
     #   return res
        
    if adata.budget >= adata.expert_cost:
        n = len(adata.taken_items)
        i = adata.taken_items[n - 1] # index of last item
        if turk_uncer[i][0] == 0 or turk_uncer[i][1] == 0: return res
        return adata.query_expert_fix(n-1)
    
    return res
    
    
    
def al_crowd_fin_expert(adata, clf, turk_uncer, crowd_budget = 5*1500):
    if adata.spent_crowd < crowd_budget and len(adata.rel) > 0:
        res = al_just_crowd(adata, clf)
        if res != -1: return res
        
    print "q expert"
    n = len(adata.taken_items)
    crowd_prob = np.zeros(n)
    found = False
    for i in range(n):
        if not adata.expert_fixed[i]:
            found = True
            j = adata.taken_items[i]
            crowd_prob[i] = turk_uncer[j][0] *1.0/ (turk_uncer[j][0] + turk_uncer[j][1])
        else:
            crowd_prob[i] = 100
            
    if not found: return -1
        
    uncertain = np.abs(crowd_prob - 0.5)
    i = np.argmin(uncertain)
    
    #print i, adata.expert_fixed[i]
    print "most", turk_uncer[adata.taken_items[i]]
    return adata.query_expert_fix(i)
    
    
score_print = []
    
def experi_money(mat, gold_rel, turk_rel, turk_uncer, n_runs = 100, money = (100*100, 1, 100), stra = 'just_crowd', sep_val = None, sep_test = None, rloss = 10, seed_size = 100):
    
    sep_val = len(gold_rel)
    sep_test = len(gold_rel)
    
    budget = money[0]; crowd_cost = money[1]; expert_cost = money[2]
    
    record = []
    record2 = []
    adatas = []
    #Run
    for run in range(n_runs):
        print run
        clf = SGDClassifier(loss="log", penalty="l1")
        adata = active_data(mat[:sep_val,:], turk_rel[:sep_val], gold_rel[:sep_val], turk_uncer, money=money)
        
        #init for voi:
        global last_current_risk
        last_current_risk = -1
        global reduction
        reduction = {'crowd_new': [], 'expert': [], 'crowd_again':[], 'crowd_all':[]}
        global last_dec
        last_dec = "none"
        global avg_reduction 
        avg_reduction= {'crowd_new': 0, 'expert': 0, 'crowd_again':0, 'crowd_all':0}
        global score_print 
        score_print = []
        
        #seed set
        #adata.budget += 2*adata.expert_cost
        #adata.query_expert_direct(7)
        #adata.query_expert_direct(0)
        saved_budget = adata.budget
        for i in range(seed_size): 
            adata.query_crowd_all(np.random.randint(adata.rel.shape[0]))
            
        adata.budget = saved_budget
        spent = []
        lauc = []
        lpen = []
        
        while True:
        
            if stra == 'jc':
                res = al_just_crowd(adata, clf)
            elif stra == 'je':
                res = al_just_expert(adata, clf)
            elif stra == 'cde':
                res = al_crowd_if_dis_expert(adata, clf, turk_uncer)
            elif stra == 'cfe':
                res = al_crowd_fin_expert(adata, clf, turk_uncer)
            elif stra == 'dec5':
                res = dec_theory(adata, clf, turk_uncer, r = [[0,1],[rloss,0]], crowd_type = 'all5', PAST = 0.0) # crowd = 5 labels
            elif stra == 'dec':
                res = dec_theory(adata, clf, turk_uncer, r = [[0,1],[rloss,0]], PAST = 0.0) # individual crowd
            elif stra == 'dec5_ip':
                res = dec_theory(adata, clf, turk_uncer, r = [[0,1],[rloss,0]], crowd_type = 'all5', PAST = 0.5)
            else:
                print 'no stra', stra
                return
                
            if adata.taken_both_classes():
                clf.fit(adata.taken_mat, adata.taken_rel)
                #auc = util.eval_clf(gold_rel[sep_test:], clf, mat[sep_test:,:])
                spent.append(budget - adata.budget)
                #lauc.append(auc)
                
                #print len(adata.taken_items), sum(adata.expert_fixed), util.eval_clf(gold_rel[sep:], clf, mat[sep:,:]), adata.evaluate(clf.predict(adata.mat))

                if stra == 'dec5_ip' or stra == 'dec5':
                    pp2 = clf.predict_proba(adata.taken_mat)
                    set_eval_rel(adata, pp2, r = [[0,1],[rloss,0]])
                penalty = adata.evaluate(clf, r = [[0,1],[rloss,0]])
                lpen.append(penalty)
                
                #dic_spent_risk[budget - adata.budget].append(risk)
            #if adata.budget % 1000 == 0:
            #    print adata.budget
                
                print penalty, sum(adata.taken_rel), len(adata.taken_rel) - sum(adata.taken_rel), budget - adata.budget, np.sum(adata.expert_fixed)
                logging.info("eval res = {0} {1} {2} {3} {4}".format(penalty, sum(adata.taken_rel), len(adata.taken_rel) - sum(adata.taken_rel), budget - adata.budget, np.sum(adata.expert_fixed)))

            if (res == -1) or (adata.mat.shape[0] <= 1):
                #record.append((spent, lauc))
                record2.append((spent, lpen))
                adatas.append(adata)
                break
            
            
            
    # Plot:

    return (record2, adatas)
    
    
def plot(record, budget = 1000, y_len = 1):
    plt.ylim([0.0, y_len*1.0])
    
    for i in range(len(record)):
        spent = record[i][0]
        auc =   record[i][1]        
        plt.plot(spent, auc, color = 'black')
    
#def plot_fill(dic_spent_auc, budget = 5000, y_len = 100):
#    X = []
#    Y = []
#    Y_std = []
#    for i in range(budget+1):
#        l = dic_spent_auc[i] 
#        if  l != []:
#            X.append(i)
#            Y.append( np.mean(l) )
#            Y_std.append( np.std(l) )
#            
#    X = np.asarray(X); Y = np.asarray(Y); Y_std = np.asarray(Y_std)
#    plt.ylim([0.0, y_len*1.0])
#    plt.plot(X, Y, color = 'black')
#    plt.fill_between(X, Y - Y_std, Y + Y_std, color = 'gray')

def plot_fill(record, lab, cl = 'red', ls = '-', marker = '.', axis = [0, 100000, 0, 2600], smooth_const = 200, ee = 50, max_spent = 100000):

    #plt.ylim([0.0, 3000])
    #plt.xlim([0.0, 100000])
    #plt.axes(axes_param)
    
    plt.axis(axis)
    data = {}
    for re in record:
        spent = re[0]
        loss  =  re[1]  
        # add the tail:
        if spent[len(spent)-1] < max_spent:
            #spent.append(max_spent)
            #print len(spent), spent[len(spent)-1]
            spent = spent + range(spent[len(spent)-1], max_spent, 100)
            loss = loss + [loss[ len(loss)-1 ]]*(len(spent) - len(loss))
            #print len(spent)
            
        for i in range(len(spent)):
            x = (spent[i]/smooth_const)*smooth_const
            y = loss[i]
            if not data.has_key(x): data[x] = [y]
            else: data[x].append(y)
        
        
    X = []; Y = []; Z = []
    for x in sorted(data.keys()):
        a = np.asarray(data[x])
        m = np.mean(a)
        s = np.std(a)
        
        X.append(x)
        Y.append(m)
        Z.append(s)
        
    X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z);  
    #plt.plot(X, Y, lw = 2, label = lab, color = cl, ls = ls)
    
    plt.errorbar(X, Y, label = lab, color = cl, ls = ls, marker = marker, yerr = Z, errorevery = ee, markevery = 30)
    
    #plt.errorbar(X, Y, label = lab, color = cl, ls = ls, marker = marker, yerr = Z)
    
    #print X[:10], Y[:10], Z[:10], (Y - Z)[:10], (Y + Z)[:10]
    #plt.fill_between(X, Y - Z, Y + Z, facecolor = cl, alpha = 0.5)
    #return (Y - Z, Y + Z)
    
########################################################################
#Value of Information

def top_uncer_items(adata, pp, n, flag = None):
    """
    Return top a flag list of top n uncertain item that not flag
    """
    uncertain = np.abs(pp[:,0] - 0.5)
    
    if flag != None:
        addition = np.asarray(flag, dtype = int)*10# flagged items are not consider, increase their value
        uncertain = uncertain + addition
    
    if len(uncertain) <= n:
        return np.nonzero(uncertain <= 10000000)[0]
    
    sorted_uncertain = np.sort(uncertain)
    
    thresh = sorted_uncertain[n]
    return np.nonzero(uncertain <= thresh)[0]


def undersam(mat, rel):
    """
    discard examples from the marjority class until
    have a balanced dataset
    """
    s = sum(rel)
    if s == 0 or s == len(rel):
        return (mat, rel)
    return util.get_balance_data(mat, rel)


def get_unl_risk(adata, pp1, i, li, r, pp3 = None):
    """
    return total risk of unlabelled items
    ignore items i
    """
    res = 0
    #pp = clf.predict_proba(adata.mat)
    # for unlabelled
    pp = pp1
    #for j in range(adata.mat.shape[0]):
    #    if j != i:
    #        res += (r[0][1] + r[1][0]) * pp[j][0] * pp[j][1]
            #res += -pp[j][0]*np.log(pp[j][0]) -pp[j][1]*np.log(pp[j][1])
            
    #optimized:
    #pp_prod = np.exp(-10*(pp[:,0] - 0.5)*(pp[:,0] - 0.5))
    #pp_prod = pp[:,0] * pp[:,1]
    
    
    pp_label = pp1
    pp_prob = pp1
    
    if pp3 != None:
        pp_label = pp3
    
    pp_prod = np.min(pp_prob, 1)
    #for i in range(len(pp_prod)):
    #    if pp_prod[i] < 0.05:
    #        pp_prod[i] = 0
            
    predict0 = pp_label[:,0] >= 0.5
    temp = np.sum(pp_prod * predict0 * r[1][0]) + np.sum(pp_prod*(1-predict0) * r[0][1])
    #print temp
    if i != -1:
        sub_for_i = pp_prod[i] * predict0[i] * r[1][0] + pp_prod[i] * (1-predict0[i]) * r[0][1]
    else:
        sub_for_i = 0
        
    res += temp - sub_for_i    
    return res
    
def set_eval_rel(adata, pp2, r):
    """
    Fix the eval_rel that minimize expected loss
    """
    return #not used
    adata.eval_rel =np.zeros(len(adata.taken_rel), dtype = int)
    for j in range(adata.taken_mat.shape[0]):
        #l = adata.taken_rel[j]
        adata.eval_rel[j] = adata.taken_rel[j]
        if not adata.expert_fixed[j]:
            k = adata.taken_items[j]
            p_true = prob_from_crowd_label2(j, adata, pp2[j])
            risk_0 = p_true[1] * r[1][0]
            risk_1 = p_true[0] * r[0][1]
            if risk_0 < risk_1: adata.eval_rel[j] = 0
            else: adata.eval_rel[j] = 1
    
def get_crowd_risk(adata, pp2, turk_uncer, r, i = -1):
    """
    risk of crowd-labelled items
    ignore item i
    pp2 = predicted_proba of labelled1166 52 76 630 2 items
    
    """
    set_eval_rel(adata, pp2, r)
    crowd_risk = 0
    for j in range(adata.taken_mat.shape[0]):
        #l = adata.eval_rel[j]
        l = adata.taken_rel[j]
        if not adata.expert_fixed[j] and j != i:
            k = adata.taken_items[j]
            p_true = prob_from_crowd_label2(j, adata, pp2[j])
            crowd_risk += p_true[1-l] * r[1-l][l]
            
    return crowd_risk
    
    
def get_risk1(adata, pp1, pp2, i, li, turk_uncer, r, crowd_type = 'single', pp3 = None):
    """
    Risk after doing action 1 (query the crowd)
    with item i and get label li
    """
    unlab_risk = get_unl_risk(adata, pp1, i, li, r = r, pp3 = pp3)
    
    #for item i
    if crowd_type == 'all5':
        # generate 5 crowd labels from the crowd accuracy model:
        # assume li is the true label 
        # eval the prob that 0,1,2,..5 crowd mems say li
        for i in range(2): 
            for j in range(2): 
                pr[i][j] = adata.count[i][j]  * 1.0 / (adata.count[i][j] + adata.count[1-i][j])
        i_risk = 0
        for x in range(6):
            p = binom.pmf(x, 5, pr[li][li])
            new_labels = [li]*x
            p_true = prob_from_crowd_label2(i, adata, pp = None, labels = new_labels)
            this_i_risk = p_true[1-li] * r[1-li][li]
            i_risk += p * this_i_risk
    else:
        new_labels = [li]
        p_true = prob_from_crowd_label2(i, adata, pp = None, labels = new_labels)
        i_risk = p_true[1-li] * r[1-li][li]
    
    #res += r[0][1] * pp[i][1] * (li == 0) \
    #    +  r[1][0] * pp[i][0] * (li == 1)
     
    # for labelled items
    #pp = pp2
    crowd_risk = get_crowd_risk(adata, pp2, turk_uncer, r = r) 
    return unlab_risk + i_risk + crowd_risk
    
    
def prob_from_crowd_label2(k, adata, pp = None, labels = None):
    """
    prob the true label given the crowd labels
    if labels not given they are labels of taken item k
    """
    #pr[i][j] = prob that crowd gives label i given true label is j 
    pr = [[0,0], [0,0]]
    for i in range(2):
        for  j in range(2):
            pr[i][j] = adata.count[i][j]  * 1.0 / (adata.count[i][j] + adata.count[1-i][j])
    
    #print pr
            
    ll = [1,1]
    if labels == None:
        labels = adata.taken_crowd_labels[k][0:adata.label_status[k]]
        
    for j in range(2):
        for i in labels:
            ll[j] = ll[j] * pr[i][j]
        
    #print ll
    
    p_true0 = ll[0] * 1.0 / (ll[0] + ll[1])
    
    #interpolate w classifier prediction
    if pp != None: p_true0 = 0.5*p_true0 + 0.5*pp[0]
    
    # unanimous bonus:
    #if uncer == (5,0):
    #    p_true0 = 0.1*p_true0 + 0.9*1.0
    
    return (p_true0, 1 - p_true0)
        
def prob_from_crowd_label(l, uncer):
    """
    return the prob of true label
    from the crowd label and workers agreements
    """
    agree = uncer[0] * 1.0 / (uncer[0] + uncer[1])
    if l == 0:
        if uncer[0] == 5: return (0.99, 0.01)
        elif uncer[0] == 4: return (0.95, 0.05)
        else: return (0.87, 0.13)
    else:
        if uncer[1] == 5: return (0.2, 0.8)
        elif uncer[1] == 4: return (0.36, 0.64)
        else: return (0.67, 0.33)

# global var to record reduction in risk
last_current_risk = -1
reduction = {'crowd_new': [], 'expert': [], 'crowd_again':[], 'crowd_all':[]}
last_dec = "none"
avg_reduction= {'crowd_new': 0, 'expert': 0, 'crowd_again':0, 'crowd_all':0}


        
        
#################################################################################
# Individual Crowd Label

def dec_theory(adata, clf, turk_uncer, r, thresh_random = 3, just_crowd = False, crowd_type = 'single', backoff= False, PAST = 0):

    n = len(adata.taken_items)        # examples taken
    m = adata.mat.shape[0]          # examples available
    if n < thresh_random or not adata.taken_both_classes():
        i = random.randint(0, m-1)
        if crowd_type == 'all5': return adata.query_crowd_all(i)
        return adata.query_crowd_single(i)
        #return adata.query_crowd(i)
        
        
    #print adata.sample_weight

    clf.partial_fit(adata.taken_mat, adata.taken_rel, classes = [0,1], sample_weight = adata.sample_weight)
    pp_unl = clf.predict_proba(adata.mat)
    pp_tak = clf.predict_proba(adata.taken_mat)
    
    
    
    #If Active Data is imbalanced ... back off
    if backoff:
        n1 = np.sum(adata.taken_rel) # examples in class 1
        if n > 10 and n < 100 and n1 * 1.0 / n < 1.0/3:
            x = np.random.rand()
            if x > 0.5:
                print 'back off'
                uncertain = np.abs(pp_unl[:,0] - 0.5)
                i = np.argmin(uncertain)
                global last_dec
                last_dec = 'crowd_new'
                if crowd_type == 'all5':
                    return adata.query_crowd_all( i, sum(uncertain)*1.0/uncertain[i] )
                return adata.query_crowd_single( i, sum(uncertain)*1.0/uncertain[i] )
    
    unlab_risk = get_unl_risk(adata, pp_unl, i = -1, li = -1, r = r)
    crowd_risk = get_crowd_risk(adata, pp_tak, turk_uncer, r = r)
    current_risk = unlab_risk + crowd_risk
    print 'exp loss: ',  unlab_risk, crowd_risk
    logging.info("exp loss = {0} {1}".format(unlab_risk, crowd_risk))
    
    # record history of reduction in risk
    global last_current_risk, reduction, use_crowd_last, last_dec	
    if last_dec != 'none':
        current_reduction = max(last_current_risk - current_risk, -10000)
        reduction[last_dec] = map (lambda x:x*0.99, reduction[last_dec])
        reduction[last_dec].append(current_reduction)
        if len(reduction[last_dec]) > WINDOW_SIZE: del reduction[last_dec][0]
        if len(reduction[last_dec]) > 0:
            avg_reduction[last_dec] = max( sum(reduction[last_dec]) * 1.0 / len(reduction[last_dec]), 0)
    
    last_current_risk = current_risk
    
    print "past: ", avg_reduction['crowd_new'], avg_reduction['expert']
    logging.info( "past: {0} {1} ".format(avg_reduction['crowd_new'], avg_reduction['expert']))
    #print (unlab_risk, crowd_risk, current_risk)
    
    eps = 0.001# minimum risk reduction
##########################################################################################
    #Action 1: Query The Crowd a unlabeled item
    # only consider top 100 items to speed up
    weight_new = 1 # weight for predicte item
    #print "weight new = ", weight_new
    item_1 = top_uncer_items(adata, pp_unl, 100)# items to be consider for action 1
    score1 = []
    score_save = []
    for i in item_1:
        
        risk = 0
        
        # pretend have label for item i
        
        for l in range(2):
            new_clf = copy.deepcopy(clf)
            new_clf.partial_fit(adata.mat[i], [l], classes = [0,1], sample_weight = [weight_new])
            pp1 = new_clf.predict_proba(adata.mat)
            pp2 = new_clf.predict_proba(adata.taken_mat)
            
            #pp3 = clf.predict_proba(adata.mat)
            
            risk_l = get_risk1(adata, pp1, pp2, i, l, crowd_type, r = r)
            
            risk += pp_unl[i][l] * risk_l
            
        #val.append(risk)
        # current score = reduction in risk, interpolate w\ average reductions
        current_score = max(current_risk - risk, eps)
        current_score = (1-PAST)*current_score + PAST*avg_reduction['crowd_new']
        score1.append(current_score)
        score_save.append(current_risk - risk)


    global score_print
    score_print.append(max(score_save))        
##########################################################################################
    #Condier crowd-labeled item: Query The Expert
    # AND: get another label 
    
    COST_RATIO = adata.expert_cost*1.0/adata.crowd_cost
    if crowd_type == 'all5':
        COST_RATIO = COST_RATIO / 5.0
    #current_risk = unlab_risk + get_crowd_risk(adata, pp2, j, r = r)
    score2 = []
    score3 = []
    
    item_2 = top_uncer_items(adata, pp_tak, 100, adata.expert_fixed) #items for action 2
    
    #for j in range(adata.taken_mat.shape[0]):
    for j in item_2:
        if (not adata.expert_fixed[j]):
            l = adata.taken_rel[j]
            #l = adata.eval_rel[j]
            k = adata.taken_items[j]
            p_true = prob_from_crowd_label2(j, adata, pp_tak[j])
            #consider expert
            if True:
                risk2 = 0
                # if the crowd is right
                
                #p_crowd_right = p_true[l]
                #risk2 += p_crowd_right * (unlab_risk + get_crowd_risk(adata, pp2, turk_uncer, j), r = r)
                #risk2 = unlab_risk + get_crowd_risk(adata, pp_tak, turk_uncer, j, r = r)
            
                # if the crowd is wrong
                #new_clf = copy.deepcopy(clf)
                #new_clf.partial_fit(adata.taken_mat[j], [1-l], classes = [0,1], sample_weight = [10])
                #new_pp1 = new_clf.predict_proba(adata.mat)
                #new_unl_risk = get_unl_risk(adata, new_pp1, i = -1, li = -1, r = r)
                #new_pp2 = new_clf.predict_proba(adata.taken_mat)
                #new_crowd_risk = get_crowd_risk(adata, new_pp2, turk_uncer, j, r = r)
                        
                #risk2 += (1-p_crowd_right) * (new_unl_risk + new_crowd_risk)
                #print p_true[1-l], r[1-l][l]
                # approximate the new risk as: old risk minus risk of this item
                risk2 = unlab_risk + crowd_risk - p_true[1-l] * r[1-l][l]
                score2.append( (1-PAST)*max((current_risk - risk2)*1.0/COST_RATIO, eps) +PAST*avg_reduction['expert']*1.0/COST_RATIO )
                #score3.append(0)
            #get another label from crowd (currently disabled)
            if adata.label_status[j] < 5 and False:
                pos = adata.label_status[j]
                save_label = adata.taken_crowd_labels[j][pos]
                adata.label_status[j] += 1
                risk = 0
                for new_crowd_l in range(2):
                    # simulate new crowd label
                    adata.taken_crowd_labels[j][pos] = new_crowd_l
                    new_l = adata.majority_vote(adata.taken_crowd_labels[j][0:adata.label_status[j]])
                    new_ptrue = prob_from_crowd_label2(j, adata, pp_tak[j])
                    new_risk = unlab_risk + crowd_risk - p_true[1-l] * r[1-l][l] + new_ptrue[1-new_l] * r[1-new_l][new_l]
                    risk += p_true[new_crowd_l] * new_risk
                    
                score3.append( (1-PAST)*max(current_risk - risk, eps) + PAST*avg_reduction['crowd_again'] )
                #score2.append(0)
                #restore
                adata.label_status[j] -= 1
                adata.taken_crowd_labels[j][pos] = save_label
            else:
                score3.append(0)    
        else:
            score2.append(0)
            score3.append(0)
            
    
    score3 = [] # disable ask crowd again
    # assign probability to each decision:
    #normalize constant z
    
    # multiply score by expected total loss ratio:
    #if crowd_risk > 0:
    #    ratio = unlab_risk * 1.0 / crowd_risk
    #else:
    #    ratio = 1.0
    #score1 = map(lambda x:x*ratio, score1)
    #print len(score1)
    
    z = 1.0*sum(score1) + sum(score2) + sum(score3)
    score1 = np.asarray(score1); score2 = np.asarray(score2); score3 = np.asarray(score3); 
    
    
    
    score = np.hstack((score1/z, score2/z, score3/z))
    
    # make decision by sampling w prob
    print 'score : ',  sum(score1), sum(score2)
    logging.info("score = {0} {1}".format(sum(score1), sum(score2)))
    #np.set_printoptions(precision=3, suppress=True)
    #print score
    pos = np.nonzero(np.random.multinomial(1, score))[0][0]
    
    #random decision w some prob after complete some normal decisions
    if len(adata.taken_rel) > 110:
        #print 'random'
        if np.random.rand() < 0.1:
            pos = np.random.randint( len(score))
    
    if pos < len(score1):
        dec = 'crowd_new'
        item = item_1[pos]
    elif pos < len(score1) + len(score2):
        dec = 'expert'
        item = item_2[pos - len(score1)]
        #item = pos - len(score1)
    #else:
     #   dec = 'crowd_again'
        #item = item_2[pos - len(score1) - len(score2)]
        
    
    #crowd_risk_reduction = 0.5*(current_risk - best1_val) + 0.5*avg_reduction
    #expert_risk_reduction = current_risk - best2_val
    #print crowd_risk_reduction, expert_risk_reduction
    
    # make decision
    #if (crowd_risk_reduction * 20 + 0.01*unlab_risk > expert_risk_reduction + 0.01*crowd_risk):
	#    dec = "crowd"
    #else:
	#    dec = "expert"

    #if np.random.random() < 0.01:
	#if np.random.random() < 0.5:
	#  dec = "crowd"
	#else:
	#  dec = "expert"
    #if (unlab_risk > crowd_risk ):
    print "dec", dec
    logging.info("dec {0}".format(dec))
    last_dec = dec
    if dec == "crowd_new":
        use_crowd_last = True
        if crowd_type == 'all5':
            return adata.query_crowd_all(item, 1.0/score[pos])
        else:
            return adata.query_crowd_single(item, 1.0/score[pos])
    elif dec == 'expert':
        print "expert"
        print turk_uncer[item]
        logging.info(str(turk_uncer[item]))  
        if adata.expert_fixed[item]:
            print "WRONG--------------------------------------------------------------"
        use_crowd_last = False
        return adata.query_expert_fix(item)
    else:
        use_crowd_last = True
        return adata.query_crowd_again(item)

