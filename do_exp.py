import start
import pickle
import active
import os
import sys
import logging
import plot


#dataset = 'proton-beam'

def save_disk(directory, filename, obj):
    f = open(directory + filename, 'w')
    pickle.dump(obj, f)
    f.close()

#expert_cost = 100
#rloss = 10
#total_cost = 100000
#runs = 1

def main(dataset, expert_cost = 100, rloss = 10, total_cost = 100000, runs = 1, y_len = 2500, suf = ""):
    
    expert_cost = int(expert_cost); rloss = int(rloss); total_cost = int(total_cost); runs = int(runs)
    #print dataset, expert_cost
    directory = 'exp_' + dataset + suf + '/'
    os.makedirs(directory)
    
    logging.basicConfig(filename=directory + '/log.log', level=logging.DEBUG)
    
    start.main(dataset)
    
    (je, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'je', rloss = rloss)
    save_disk(directory, 'je.pkl', je)


    (jc, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'jc', rloss = rloss)
    save_disk(directory, 'jc.pkl', jc)


    (cde, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'cde',  rloss = rloss)
    save_disk(directory, 'cde.pkl', cde)


#(dec5, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, 5, (100000, 1, expert_cost), stra = 'dec5',  rloss = rloss)
#save_disk('dec5.pkl', dec5)



    (dec5_ip, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'dec5_ip',  rloss = rloss )
    save_disk(directory, 'dec5_ip.pkl', dec5_ip)
    
    #PLOT
    plot.main(directory, y_len)
    
if __name__ == "__main__":
   main(*sys.argv[1:])
