import re
import numpy as np
import pickle as pk
import os


def get_prob_table_pk_filename(filename):
    pickle_prefix = "pickle/prob_tables/"
    bif_suffix_len = len(".bif")
    model_id = filename[filename.rindex("_")+1:-bif_suffix_len]
    prob_table_pk_filename = pickle_prefix + model_id + "_prob_tables.pkl"
    return prob_table_pk_filename

def parse(filename):
    # open file
    with open(filename, "r") as file:
        lines = file.readlines()
    
    # stored state
    variable_names = list()
    variable_to_table = dict()

    # iterate line by line
    i = 0
    while (i < len(lines)):
        line = lines[i]
        
        # get variable names (unused)
        if ("<VARIABLE" in line):
            assert(i + 1 < len(lines))
            variable_name_line = lines[i+1]
            m = re.search("<NAME>(.*)</NAME>", variable_name_line)
            variable_names.append(m.group(1))
            i += 1

        # get probability table for given variable
        elif ("<FOR>" in line):
            # check integrity
            assert(i + 1 < len(lines))
            i += 1
            if ("<GIVEN>" in lines[i]):
                i += 1

            # build table
            m = re.search("<FOR>(.*)</FOR>", line)
            variable_name = m.group(1)
            assert(not variable_name in variable_to_table)
            variable_to_table[variable_name] = list()
            
            table_line = lines[i]
            assert("<TABLE>" in table_line)
            i += 1
            while (not "</TABLE>"in lines[i]):
                try:
                    row = np.array([float(j) for j in lines[i].split(" ") if j != "\n"], dtype=np.float32)
                    variable_to_table[variable_name].append(row)
                except:
                    print("ERROR: could not parse into floats:")
                    print(lines[i].split(" "))
                i += 1
            if (variable_name in variable_to_table):
                variable_to_table[variable_name] = np.array(variable_to_table[variable_name])
        i += 1

    # save probability tables with pickle
    prob_table_pk_filename = get_prob_table_pk_filename(filename)
    print(prob_table_pk_filename)
    os.makedirs(os.path.dirname(prob_table_pk_filename), exist_ok=True)
    with open(prob_table_pk_filename, "wb") as handle:
        pk.dump(variable_to_table, handle)

def parse_all(dirname):
    #print(os.listdir(dirname))
    for filename in os.listdir(dirname):
        if (filename.endswith(".bif")):
            parse(os.path.join(dirname, filename))

if __name__ == "__main__":
    fileLoc = "../../data/"
    parse_all(fileLoc + "source_model/") #other_source_model/
    
