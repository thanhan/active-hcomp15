from start import *
import pickle

directory = 'exp_cheap_expert/'

def save_disk(filename, obj):
    f = open(directory + filename, 'w')
    pickle.dump(obj, f)
    f.close()

expert_cost = 10
recall_loss = 10
total_cost = 50000

(je, adata) = active.experi_money(util.mat, util.rel, turk_data, turk_data_uncer, 5, (total_cost, 1, expert_cost), stra = 'je', recall_loss = recall_loss)
save_disk('je.pkl', je)


(jc, adata) = active.experi_money(util.mat, util.rel, turk_data, turk_data_uncer, 5, (total_cost, 1, expert_cost), stra = 'jc', recall_loss = recall_loss)
save_disk('jc.pkl', jc)


(cde, adata) = active.experi_money(util.mat, util.rel, turk_data, turk_data_uncer, 5, (total_cost, 1, expert_cost), stra = 'cde',  recall_loss = recall_loss)
save_disk('cde.pkl', cde)


#(dec5, adata) = active.experi_money(util.mat, util.rel, turk_data, turk_data_uncer, 5, (100000, 1, expert_cost), stra = 'dec5',  recall_loss = recall_loss)
#save_disk('dec5.pkl', dec5)



(dec5_ip, adata) = active.experi_money(util.mat, util.rel, turk_data, turk_data_uncer, 5, (total_cost, 1, expert_cost), stra = 'dec5_ip',  recall_loss = recall_loss )

save_disk('dec5_ip.pkl', dec5_ip)
