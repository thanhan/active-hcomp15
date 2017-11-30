count = [0,0]
for i, t in enumerate(turk_data_uncer):
    if t == (3,2):
        print i, t, util.rel[i]
        count[int(util.rel[i])] += 1
        
print count
print count[0]*1.0/(count[0]+count[1])
