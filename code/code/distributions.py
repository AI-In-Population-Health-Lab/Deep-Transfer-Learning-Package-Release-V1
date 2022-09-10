# import re
# import numpy as np
# import pickle as pk
# import os
# import copy

import numpy as np
import pickle as pk
from parse_model import get_prob_table_pk_filename
import os

# d_KL(p1 || p2), p1 is the target distribution, p2 is the source distribution
def get_avg_kl(filename1, filename2):
    # load tables
    prob_table_pk_filename1 = get_prob_table_pk_filename(filename1)
    prob_table_pk_filename2 = get_prob_table_pk_filename(filename2)
    
    with open(prob_table_pk_filename1, "rb") as handle:
        prob_table1 = pk.load(handle)
    with open(prob_table_pk_filename2, "rb") as handle:
        prob_table2 = pk.load(handle)
    
    # check integrity
    assert(set(prob_table1.keys()) == set(prob_table2.keys()))

    total_d_kl = 0
    max_d_kl = 0

    for var in prob_table1:
        prob_dist1 = prob_table1[var]
        prob_dist2 = prob_table2[var]

        # sum_{x} p1(x) log(p1(x) / p2(x))
        var_d_kl = 0
        num_conditions, num_features = prob_dist1.shape
        for i in range(num_conditions):
            d_kl = 0
            for j in range(num_features):
                d_kl += prob_dist1[i][j] * np.log(prob_dist1[i][j] / prob_dist2[i][j])
            var_d_kl += d_kl
        
        # print(var, "avg d_KL", var_d_kl / num_conditions)
        total_d_kl += var_d_kl / num_conditions
        max_d_kl = max(max_d_kl, var_d_kl / num_conditions)
    
    avg_d_kl = total_d_kl / len(prob_table1)
    print("overall avg d_KL", avg_d_kl)
    # print("max avg d_KL", max_d_kl)

def get_kl(filename1, filename2):
    # print(filename1, filename2)
    # load tables
    prob_table_pk_filename1 = get_prob_table_pk_filename(filename1)
    prob_table_pk_filename2 = get_prob_table_pk_filename(filename2)
    
    with open(prob_table_pk_filename1, "rb") as handle:
        prob_table1 = pk.load(handle)
    with open(prob_table_pk_filename2, "rb") as handle:
        prob_table2 = pk.load(handle)
    
    # check integrity
    assert(set(prob_table1.keys()) == set(prob_table2.keys()))

    p_d1 = prob_table1["diagnosis"][0]
    p_d2 = prob_table2["diagnosis"][0] # unused

    d_kl = 0

    for var in prob_table1:
        if (var != "diagnosis"):
            prob_dist1 = prob_table1[var]
            prob_dist2 = prob_table2[var]
            assert(prob_dist1.shape == prob_dist2.shape)
            for i in range(prob_dist1.shape[0]): # diagnosis setting
                for j in range(prob_dist1.shape[1]): # feature setting
                    p = prob_dist1[i][j]
                    q = prob_dist2[i][j]
                    d_kl += p * p_d1[i] * np.log(p / q)    

    print("d_kl", d_kl)
    return d_kl


if __name__ == "__main__":
    fileLoc ="../../data/"
    filenameLoc1 = fileLoc + "source_model/"
    filename2 = fileLoc + "target_model/findings_final_0814.bif"
    result={}
    for filename in os.listdir(filenameLoc1):
        if (filename.endswith(".bif")):
            kl= get_kl(filename2, filename)  # KL(target,source)
            result[kl] = filename
    import collections
    od = collections.OrderedDict(sorted(result.items()))
    for key in od:
        print(key,od[key])
