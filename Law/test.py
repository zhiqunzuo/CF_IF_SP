import numpy as np
import os
import pandas as pd
import random

from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

data = pd.read_csv("generated_data.csv")

for seed in range(42, 47):
    train_set, valid_set = train_test_split(data, test_size=0.2, random_state=seed)
    valid_set, test_set = train_test_split(data, test_size=0.5, random_state=seed)
    
    
    train_bare_set = train_set.drop(["K", "c_UGPA", "c_LSAT"], axis=1)
    valid_bare_set = valid_set.drop(["K", "c_UGPA", "c_LSAT"], axis=1)
    test_bare_set = test_set.drop(["K", "c_UGPA", "c_LSAT"], axis=1)
    train_bare_set["sex"] = train_bare_set["sex"] - 1
    valid_bare_set["sex"] = valid_bare_set["sex"] - 1
    test_bare_set["sex"] = test_bare_set["sex"] - 1
    train_bare_set = pd.get_dummies(train_bare_set, columns=["race"], prefix="", prefix_sep="")
    race_names = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other",
                  "Puertorican", "White"]
    train_bare_set[race_names] = train_bare_set[race_names].astype(int)
    valid_bare_set = pd.get_dummies(valid_bare_set, columns=["race"], prefix="", prefix_sep="")
    race_names = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other",
                  "Puertorican", "White"]
    valid_bare_set[race_names] = valid_bare_set[race_names].astype(int)
    test_bare_set = pd.get_dummies(test_bare_set, columns=["race"], prefix="", prefix_sep="")
    race_names = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other",
                  "Puertorican", "White"]
    test_bare_set[race_names] = test_bare_set[race_names].astype(int)
     
    train_bare_set.to_csv(os.path.join("data", "generated_train_{}.csv".format(seed)), index=False)
    valid_bare_set.to_csv(os.path.join("data", "generated_valid_{}.csv".format(seed)), index=False)
    test_bare_set.to_csv(os.path.join("data", "generated_test_{}.csv".format(seed)), index=False)
    
    train_set.to_csv(os.path.join("data", "generated_train_{}_counter.csv".format(seed)), index=False)
    valid_set.to_csv(os.path.join("data", "generated_valid_{}_counter.csv".format(seed)), index=False)
    test_set.to_csv(os.path.join("data", "generated_test_{}_counter.csv".format(seed)), index=False)