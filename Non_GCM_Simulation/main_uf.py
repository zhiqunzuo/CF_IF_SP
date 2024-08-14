import numpy as np
import os
import pandas as pd
import pickle

def generate_representation(dataset):
    dataset = pd.get_dummies(dataset, columns=["race"], prefix="", prefix_sep="")
    race_names = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other", 
                  "Puertorican", "White"]
    dataset[race_names] = dataset[race_names].astype(int)
    
    race = np.array(dataset[race_names])
    A = np.array(dataset["sex"]) - 1
    
    h = np.concatenate([A.reshape(-1, 1), race, np.array(dataset["UGPA"]).reshape(-1, 1),
                        np.array(dataset["LSAT"]).reshape(-1, 1)], axis=1)
    c_h = np.concatenate([(1 - A).reshape(-1, 1), race, np.array(dataset["c_UGPA"]).reshape(-1, 1),
                          np.array(dataset["c_LSAT"]).reshape(-1, 1)], axis=1)
    
    fya = np.array(dataset["ZFYA"])
    
    return h, c_h, A, fya

def main():
    for seed in range(42, 47):
        train_set = pd.read_csv(os.path.join("data", "generated_train_{}_counter.csv".format(seed)))
        valid_set = pd.read_csv(os.path.join("data", "generated_valid_{}_counter.csv".format(seed)))
        test_set = pd.read_csv(os.path.join("data", "generated_test_{}_counter.csv".format(seed)))
        
        train_h, train_c_h, train_a, train_y = generate_representation(train_set)
        valid_h, valid_c_h, valid_a, valid_y = generate_representation(valid_set)
        test_h, test_c_h, test_a, test_y = generate_representation(test_set)
        
        rep = {"train_h": train_h, "train_c_h": train_c_h, "train_a": train_a, "train_y": train_y,
               "valid_h": valid_h, "valid_c_h": valid_c_h, "valid_a": valid_a, "valid_y": valid_y,
               "test_h": test_h, "test_c_h": test_c_h, "test_a": test_a, "test_y": test_y}
        
        with open(os.path.join("representation", "UF_{}.pkl".format(seed)), "wb") as f:
            pickle.dump(rep, f)

if __name__ == "__main__":
    main()