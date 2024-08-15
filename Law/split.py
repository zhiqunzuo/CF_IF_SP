import numpy as np
import os
import pandas as pd
import random

from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)

race_names = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other",
              "Puertorican", "White"]

for seed in range(42, 47):
    train_data = np.load(os.path.join("final_data", "train_data_{}.npz".format(seed)))
    
    train_frame_counter = {"sex": train_data["a"][:, -1],
                           "LSAT_I": train_data["sat_i"].reshape(-1,),
                           "LSAT": train_data["sat"].reshape(-1),
                           "c_LSAT": train_data["c_sat"].reshape(-1),
                           "UGPA_I": train_data["gpa_i"].reshape(-1,),
                           "UGPA": train_data["gpa"].reshape(-1),
                           "c_UGPA": train_data["c_gpa"].reshape(-1),
                           "K": train_data["K"].reshape(-1),
                           "ZFYA": train_data["fya"].reshape(-1)
                           }
    for i in range(8):
        train_frame_counter[race_names[i]] = train_data["a"][:, i]
    
    train_frame_counter = pd.DataFrame(train_frame_counter)    
    train_frame_counter.to_csv(os.path.join("data", "law_train_{}_counter.csv".format(seed)), index=False)
    
    valid_data = np.load(os.path.join("final_data", "valid_data_{}.npz".format(seed)))
    
    valid_frame_counter = {"sex": valid_data["a"][:, -1],
                           "LSAT_I": valid_data["sat_i"].reshape(-1,),
                           "LSAT": valid_data["sat"].reshape(-1),
                           "c_LSAT": valid_data["c_sat"].reshape(-1),
                           "UGPA_I": valid_data["gpa_i"].reshape(-1,),
                           "UGPA": valid_data["gpa"].reshape(-1),
                           "c_UGPA": valid_data["c_gpa"].reshape(-1),
                           "K": valid_data["K"].reshape(-1),
                           "ZFYA": valid_data["fya"].reshape(-1),
                           }
    for i in range(8):
        valid_frame_counter[race_names[i]] = valid_data["a"][:, i]
    
    valid_frame_counter = pd.DataFrame(valid_frame_counter)    
    valid_frame_counter.to_csv(os.path.join("data", "law_valid_{}_counter.csv".format(seed)), index=False)
    
    test_data = np.load(os.path.join("final_data", "test_data_{}.npz".format(seed)))
    
    test_frame_counter = {"sex": test_data["a"][:, -1],
                          "LSAT_I": test_data["sat_i"].reshape(-1,),
                          "LSAT": test_data["sat"].reshape(-1),
                          "c_LSAT": test_data["c_sat"].reshape(-1),
                          "UGPA_I": test_data["gpa_i"].reshape(-1),
                          "UGPA": test_data["gpa"].reshape(-1),
                          "c_UGPA": test_data["c_gpa"].reshape(-1),
                          "K": test_data["K"].reshape(-1),
                          "ZFYA": test_data["fya"].reshape(-1)
                          }
    for i in range(8):
        test_frame_counter[race_names[i]] = test_data["a"][:, i]
    
    test_frame_counter = pd.DataFrame(test_frame_counter)    
    test_frame_counter.to_csv(os.path.join("data", "law_test_{}_counter.csv".format(seed)), index=False)
    
    train_frame = train_frame_counter.drop(["UGPA", "c_UGPA", "LSAT", "c_LSAT", "K"], axis=1)
    train_frame.to_csv(os.path.join("data", "law_train_{}.csv".format(seed)), index=False)
    valid_frame = valid_frame_counter.drop(["UGPA", "c_UGPA", "LSAT", "c_LSAT", "K"], axis=1)
    valid_frame.to_csv(os.path.join("data", "law_valid_{}.csv".format(seed)), index=False)
    test_frame = test_frame_counter.drop(["UGPA", "c_UGPA", "LSAT", "c_LSAT", "K"], axis=1)
    test_frame.to_csv(os.path.join("data", "law_test_{}.csv".format(seed)), index=False)