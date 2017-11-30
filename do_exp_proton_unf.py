import start
import pickle
import active
import os

dataset = 'proton-beam'

directory = 'exp_' + dataset + '_unf' + '/'
os.makedirs(directory)


start.main(dataset)

def save_disk(filename, obj):
    f = open(directory + filename, 'w')
    pickle.dump(obj, f)
    f.close()

expert_cost = 100
rloss = 10
total_cost = 100000
runs = 1


(je, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'je', rloss = rloss)
save_disk('je.pkl', je)


(jc, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'jc', rloss = rloss)
save_disk('jc.pkl', jc)


(cde, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'cde',  rloss = rloss)
save_disk('cde.pkl', cde)


#(dec5, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, 5, (100000, 1, expert_cost), stra = 'dec5',  rloss = rloss)
#save_disk('dec5.pkl', dec5)



(dec5_ip, adata) = active.experi_money(start.mat, start.rel, start.turk_data, start.turk_data_uncer, runs, (total_cost, 1, expert_cost), stra = 'dec5_ip',  rloss = rloss )

save_disk('dec5_ip.pkl', dec5_ip)
