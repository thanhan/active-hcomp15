import util

import active
import numpy as np
import util2

mat = None
rel = None
turk_data = None
turk_data_uncer = None

def main(dataset = 'proton-beam'):
    global mat, rel, turk_data, turk_data_uncer
    
    util.main(dataset)
    mat = util.mat
    rel = util.rel
    
    util2.main(dataset, util.turk_dic)
    turk_data = util2.turk_data
    turk_data_uncer = util2.turk_data_uncer
    





if __name__ == "__main__":
    main()