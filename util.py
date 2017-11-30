import xml.etree.ElementTree as ET
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import scipy

import csv

def get_text(a):
    try:
        return a.text
    except AttributeError:
        return ''
        
def get_relevant():
    f = open('data/proton-beam-relevant.txt')
    res = np.zeros(4751)
    for line in f:
        x = int(line)
        res[x-1] = 1
    f.close()
    return res
    
def get_pub_dic_xml(file_name = 'data/proton-beam-all.xml'):
    tree = ET.parse(file_name)
    root = tree.getroot()[0]

    # Create dic of : id -> feature text
    pub_dic = {}
    for pub in root:
        rec_number = int (get_text (pub.find('rec-number')))
        abstract   = get_text (pub.find('abstract'))
        title      = get_text (pub.find('titles')[0])
        text = title + abstract
        for kw in pub.find('keywords'):
            text = text + kw.text + ' '
        pub_dic[rec_number] = text
        
    return pub_dic
        
        


def get_pub_dic_csv(dataset):
    filename = "data/" + dataset + "-text.csv"
    f = open(filename)
    f.readline()
    csv_reader = csv.reader(f)
    
    # Create dic of : id -> feature text
    pub_dic = {}
    
    for row in csv_reader:        
        (abstract_id, title, publisher, abstract) = tuple(row)[0:4]
        abstract_id = int(abstract_id)
        text = title + abstract
        
        pub_dic[abstract_id] = text
        
    return pub_dic
    
    
def get_turk_data(dataset):
    filename = "data/" + dataset + "-turk.csv"
    f = open(filename)
    first_line = f.readline()
    csv_reader = csv.reader(f)
    
    turk_dic = {}
    rel_dic = {}
    for row in csv_reader:
        #print len(row)
        if dataset == 'omega3':
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId, Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)
        else:
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId, Question1, Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)
        AbstractId = int(AbstractId)
        if AbstractId not in turk_dic: turk_dic[AbstractId] = []
        turk_dic[AbstractId].append( (Question3, Question4) )
        rel_dic[AbstractId] = Relevant
        
    return (turk_dic, rel_dic)

    
mat = None
rel = None
turk_dic = None
    
def main(dataset = 'proton-beam'):
    global mat, rel, turk_dic
    
    if dataset == 'proton-beam':
        pub_dic = get_pub_dic_xml()    
        # pub_dic_items are already sorted by key
        [rec_nums, texts] = zip(*pub_dic.items())
        rel = get_relevant()
    else:
        pub_dic = get_pub_dic_csv(dataset)
        #[rec_nums, texts] = zip(*pub_dic.items())
        (turk_dic, rel_dic) = get_turk_data(dataset)
        texts = []
        for i in pub_dic.keys():
            if pub_dic.has_key(i) and turk_dic.has_key(i) and rel_dic.has_key(i):
                texts.append(pub_dic[i])
            else:
                if pub_dic.has_key(i): pub_dic.pop(i)
                if turk_dic.has_key(i): turk_dic.pop(i)
                if rel_dic.has_key(i): rel_dic.pop(i)
                
        (_,rel) = zip(*rel_dic.items())
        rel = map(int, rel)
        
    vectorizer = TfidfVectorizer()
    #save_texts = texts
    mat = vectorizer.fit_transform(texts)
    return (pub_dic, texts)


def classify(n = 50):
    #clf = MultinomialNB(fit_prior=False)
    #clf = SVC(gamma=2, C=1, class_weight = {0.0:0.063829777, 1.0:1.0})
    clf = SGDClassifier(loss="log", penalty="l1", class_weight = {0.0:0.022, 1.0:1.0})

    clf.fit(mat[:n], rel[:n])
    return clf

    
def confu_mat(rel, turk_rel):
    m = [[0,0],[0,0]]
    for i in range(len(rel)):
        m[rel[i]][turk_rel[i]] += 1
    return m
    
def plot_pr(gold, predicted_prob, lb):
    pp1 = predicted_prob[:,1] # prob for class 1
    p, r, th = precision_recall_curve(gold, pp1)
    ap = average_precision_score(gold, pp1)
    plt.plot(r, p, label= lb + ' (area = {0:0.2f})'
                   ''.format(ap))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision and Recall')
    plt.legend(loc="upper right")
    #plt.show()
    
def eval_clf(gold, clf, mat, start = 0):
    pp = clf.predict_proba(mat[start:,:])
    pp1 = pp[:,1]
    ap = average_precision_score(gold[start:], pp1)
    return ap

def train_and_plot(ex = [50,100,200]):
    """
    train the classifier with ex[i] examples
    Plot
    """
    
    for num in ex:
        clf = classify(num)
        pp = clf.predict_proba(mat)
        plot_pr(rel[2000:], pp[2000:], str(num))
        
        
def get_balance_data(mat, rel):
    mat_1 = mat[ np.nonzero(rel == 1)[0] ]
    mat_0 = mat[ np.nonzero(rel == 0)[0] ]
    
    #print mat_1.shape, mat_0.shape

    n = min(mat_1.shape[0], mat_0.shape[0])
    
    #shuffle mat_0
    index = np.arange( mat_0.shape[0] )
    np.random.shuffle(index)
    mat_0 = mat_0[index]
    
    #print mat_0.shape
    
    new_mat = scipy.sparse.vstack([mat_1[:n], mat_0[:n]], 'csr')
    new_rel = np.hstack([np.ones((n,)), np.zeros((n,))] )
    
    #print new_mat, new_rel.shape
    
    #shuffle new mat and rel
    index = np.arange(new_mat.shape[0])
    np.random.shuffle(index)
    
    new_mat = new_mat[index]
    new_rel = new_rel[index]
    
    return (new_mat, new_rel)
    

    #s = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20, 37, 44, 68, 71, 118, 141, 162,183, 189, 248, 249, 255, 267, 268, 324]
    
    #
    #from sklearn.cross_validation import KFold
    #kf = KFold(n, n_folds=10)
    #acc_list = []
    #for train, test in kf:
    #    clf.fit(mat[train], rel[train])
    #    predicted = clf.predict(mat[test])
    #    acc = sum(predicted == rel[test]) * 1.0 / len(rel[test])
    #    acc_list.append(acc)
        
    #print 'average accuracy: ', np.average(acc_list)

    #for i in range(20, 1000, 20):
    #    clf.fit(mat[0:i], rel[0:i])
    #    predicted = clf.predict(mat[1000:])
    #    acc = sum(predicted == rel[1000:]) * 1.0 / len(rel[1000:])
    #    print i, acc
    #from sklearn.svm import SVC

    #clf = SVC()

    #clf.fit(mat, rel)

    
