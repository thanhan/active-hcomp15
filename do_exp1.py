from start import *
import pickle

def save_disk(filename, obj):
    f = open(filename, 'w')
    pickle.dump(obj, f)
    f.close()


(dec5, adata) = active.experi_money(util.mat, util.rel, turk_data, turk_data_uncer, 5, (100000, 1, 100), stra = 'dec5_ip')
save_disk('dec5_old_estimator.pkl', dec5)
