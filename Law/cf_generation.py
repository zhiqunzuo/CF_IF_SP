import argparse
import copy
import numpy as np
import os
import pandas as pd
import pickle
import random
import stan

from pathlib import Path
from sklearn.model_selection import train_test_split

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_data_preprocess(seed, data_path):
    law_data = pd.read_csv(data_path, index_col=0)
    law_data = pd.get_dummies(law_data, columns=["race"], prefix="", prefix_sep="")
    
    try:
        law_data = law_data.drop(["region_first", "sander_index", "first_pf"], axis=1)
    except:
        pass
    
    law_data["male"] = law_data["sex"].map(lambda z: 1 if z == 2 else 0)
    law_data["female"] = law_data["sex"].map(lambda z: 1 if z == 1 else 0)
    
    #law_data["LSAT"] = law_data["LSAT"].apply(lambda x: int(np.round(x)))
    
    law_data = law_data.drop(axis=1, columns=["sex"])
    
    sense_cols = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other",
                  "Puertorican", "White", "male", "female"]
    
    law_data[sense_cols] = law_data[sense_cols].astype(int)
    
    law_train, law_valid_test = train_test_split(law_data, random_state=seed, test_size=0.2)
    law_valid, law_test = train_test_split(law_valid_test, random_state=seed, test_size=0.5)
    
    law_train.to_csv(os.path.join(data_path[:-4], "train_{}.csv".format(seed)), index=False)
    law_valid.to_csv(os.path.join(data_path[:-4], "valid_{}.csv".format(seed)), index=False)
    law_test.to_csv(os.path.join(data_path[:-4], "test_{}.csv".format(seed)), index=False)
    
    return law_train, law_valid, law_test, sense_cols

def get_inference_dict(pandas_df, sense_cols):
    dic_out = {}
    dic_out["N"] = len(pandas_df)
    dic_out["K"] = len(sense_cols)
    dic_out["a"] = np.array(pandas_df[sense_cols])
    dic_out["ugpa"] = list(pandas_df["UGPA"])
    #dic_out["lsat"] = list(pandas_df["LSAT"].astype(int))
    dic_out["lsat"] = list(pandas_df["LSAT"])
    dic_out["zfya"] = list(pandas_df["ZFYA"])
    return dic_out

def get_abduction_dict(data_dic, post_samps):
    dic_out = {}
    for key in post_samps.keys():
        if key not in ["sigma_g_Sq_gpa", "sigma_g_Sq_sat", "sigma_g_Sq_fya", "u", "zfya0", "eta_a_zfya", "eta_u_zfya"]:
            dic_out[key] = np.mean(post_samps[key], axis=1)
            if dic_out[key].size == 1:
                dic_out[key] = dic_out[key].item()
    
    need_list = ["N", "K", "a", "ugpa", "lsat", "zfya"]
    for data in need_list:
        dic_out[data] = data_dic[data]
    
    return dic_out

def inference_preprocess(law_train, law_test, sense_cols):
    law_train_dic = get_inference_dict(law_train, sense_cols)
    law_test_dic = get_inference_dict(law_test, sense_cols)
    return law_train_dic, law_test_dic

def inference(law_train_dic, data_path, seed, dataset):
    result_path = "./saved_models/inference_result_{}".format(seed)
    check_fit = Path(result_path)
    
    if check_fit.is_file():
        print("File Found: Loading Fitted Inference Model...")
        with open(result_path, "rb") as f:
            post_samps = pickle.load(f)
    else:
        if dataset == "law":
            with open("law_school_train.stan", "r") as f:
                model_code = f.read()
        else:
            with open("generated_train.stan", "r") as f:
                model_code = f.read()
        print("model code =\n", model_code)
        model = stan.build(model_code, data=law_train_dic, random_seed=seed)
        print("Finished compiling model!")
        fit = model.sample(num_chains=4, num_samples=1000)
        post_samps = fit
        
        with open(result_path, "wb") as f:
            pickle.dump(post_samps, f, protocol=-1)
            print("Saved fitted model!")
    
    return post_samps

def abduction_preprocess(law_train_dic, law_test_dic, parameter_dic):
    law_train_dic_final = get_abduction_dict(law_train_dic, parameter_dic)
    law_test_dic_final = get_abduction_dict(law_test_dic, parameter_dic)
    return law_train_dic_final, law_test_dic_final

def abduction(final_dic, split, seed, data_path, dataset):
    result_path = "./saved_models/abduction_{}_{}".format(split, seed)
    check_fit = Path(result_path)
    
    if check_fit.is_file():
        print("File Found: {} knowledge exists".format(split))
        with open(result_path.format(split, seed), "rb") as f:
            K_samples = pickle.load(f)
    else:
        if dataset == "law":
            with open("law_school_u.stan", "r") as f:
                model_code = f.read()
        else:
            with open("generated_u.stan", "rb") as f:
                model_code = f.read()
        print("model_code =\n", model_code)
        model = stan.build(model_code, data=final_dic, random_seed=seed)
        print("Finished compiling model!")
        fit = model.sample(num_chains=4, num_samples=1000)
        K_samples = fit
        
        with open(result_path, "wb") as f:
            pickle.dump(K_samples, f, protocol=-1)
            print("Saved fitted model!")
    
    #K = np.stack([K_samples["u1"], K_samples["u2"], K_samples["u3"], K_samples["u4"]], axis=2)
    K = K_samples["u"]
    return K

def get_gpa(k, final_dic):
    gpa = np.empty((final_dic["N"], 1))
    c_gpa = np.empty((final_dic["N"], 1))
    for i in range(final_dic["N"]):
        #gpa_samps = final_dic["ugpa0"] + final_dic["eta_u1_ugpa"] * k[i, :, 0] + \
        #    final_dic["eta_u2_ugpa"] * k[i, :, 1] + final_dic["eta_u4_ugpa"] * k[i, :, 3] + \
        #    np.dot(final_dic["eta_a_ugpa"], final_dic["a"][i])
        gpa_samps = final_dic["ugpa0"] + final_dic["eta_u_ugpa"] * k[i] + \
            np.dot(final_dic["eta_a_ugpa"], final_dic["a"][i])
        c_a = final_dic["a"][i]
        c_a[-2:] = 1 - c_a[-2:]
        #c_gpa_samps = final_dic["ugpa0"] + final_dic["eta_u1_ugpa"] * k[i, :, 0] + \
        #    final_dic["eta_u2_ugpa"] * k[i, :, 1] + final_dic["eta_u4_ugpa"] * k[i, :, 3] + \
        #    np.dot(final_dic["eta_a_ugpa"], c_a)
        c_gpa_samps = final_dic["ugpa0"] + final_dic["eta_u_ugpa"] * k[i] + \
            np.dot(final_dic["eta_a_ugpa"], c_a)
        gpa[i] = np.mean(gpa_samps)
        c_gpa[i] = np.mean(c_gpa_samps)
    return gpa, c_gpa

def get_sat(k, final_dic):
    sat = np.empty((final_dic["N"], 1))
    c_sat = np.empty((final_dic["N"], 1))
    for i in range(final_dic["N"]):
        #sat_samps = np.exp(final_dic["lsat0"] + final_dic["eta_u1_lsat"] * k[i, :, 0] + \
        #    final_dic["eta_u3_lsat"] * k[i, :, 2] + final_dic["eta_u4_lsat"] * k[i, :, 3] + \
        #    np.dot(final_dic["eta_a_lsat"], final_dic["a"][i]))
        sat_samps = np.exp(final_dic["lsat0"] + final_dic["eta_u_lsat"] * k[i] + \
            np.dot(final_dic["eta_a_lsat"], final_dic["a"][i]))
        c_a = final_dic["a"][i]
        c_a[-2:] = 1 - c_a[-2:]
        #c_sat_samps = np.exp(final_dic["lsat0"] + final_dic["eta_u1_lsat"] * k[i, :, 0] + \
        #    final_dic["eta_u3_lsat"] * k[i, :, 2] + final_dic["eta_u4_lsat"] * k[i, :, 3] + \
        #    np.dot(final_dic["eta_a_lsat"], c_a))
        c_sat_samps = np.exp(final_dic["lsat0"] + final_dic["eta_u_lsat"] * k[i] + \
            np.dot(final_dic["eta_a_lsat"], c_a))
        sat[i] = np.mean(sat_samps)
        c_sat[i] = np.mean(c_sat_samps)
    print("sat = {}".format(sat))
    return sat, c_sat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="./data/law_data.csv")
    parser.add_argument("--dataset", type=str, choices=["law", "generated"], default="generated")
    args = parser.parse_args()
    law_train, law_valid, law_test, sense_cols = get_data_preprocess(args.seed, args.data_path)
    law_train_valid = pd.concat([law_train, law_valid])
    law_train_valid_dic, law_test_dic = inference_preprocess(law_train_valid, law_test, sense_cols)
    parameter_dic = inference(law_train_valid_dic, args.data_path, args.seed, args.dataset)
    law_train_valid_dic_final, law_test_dic_final = abduction_preprocess(law_train_valid_dic, 
                                                                   law_test_dic, parameter_dic)
    law_train_dic_final = copy.deepcopy(law_train_valid_dic_final)
    law_valid_dic_final = copy.deepcopy(law_train_valid_dic_final)
    law_train_dic_final["ugpa"], law_train_dic_final["lsat"], law_train_dic_final["a"], law_train_dic_final["zfya"] = \
        law_train_dic_final["ugpa"][:len(law_train)], law_train_dic_final["lsat"][:len(law_train)], law_train_dic_final["a"][:len(law_train)], law_train_dic_final["zfya"][:len(law_train)]
    law_valid_dic_final["ugpa"], law_valid_dic_final["lsat"], law_valid_dic_final["a"], law_valid_dic_final["zfya"] = \
        law_valid_dic_final["ugpa"][len(law_train):], law_valid_dic_final["lsat"][len(law_train):], law_valid_dic_final["a"][len(law_train):], law_valid_dic_final["zfya"][len(law_train):]
    law_train_dic_final["N"] = len(law_train)
    law_valid_dic_final["N"] = len(law_valid)
    
    train_K = abduction(law_train_dic_final, "train", args.seed, args.data_path, args.dataset)
    valid_K = abduction(law_valid_dic_final, "valid", args.seed, args.data_path, args.dataset)
    test_K = abduction(law_test_dic_final, "test", args.seed, args.data_path, args.dataset)
    
    train_gpa, train_c_gpa = get_gpa(train_K, law_train_dic_final)
    train_sat, train_c_sat = get_sat(train_K, law_train_dic_final)
    
    valid_gpa, valid_c_gpa = get_gpa(valid_K, law_valid_dic_final)
    valid_sat, valid_c_sat = get_sat(valid_K, law_valid_dic_final)
    
    test_gpa, test_c_gpa = get_gpa(test_K, law_test_dic_final)
    test_sat, test_c_sat = get_sat(test_K, law_test_dic_final)
    print("test K shape = {}".format(test_K.shape))
    
    train_data = {"K": np.mean(train_K, axis=1), "gpa": train_gpa, "c_gpa": train_c_gpa,
                  "gpa_i": law_train_dic_final["ugpa"], "sat": train_sat, "c_sat": train_c_sat,
                  "sat_i": law_train_dic_final["lsat"], "fya": law_train_dic_final["zfya"],
                  "a": law_train_dic_final["a"]}
    valid_data = {"K": np.mean(valid_K, axis=1), "gpa": valid_gpa, "c_gpa": valid_c_gpa,
                  "gpa_i": law_valid_dic_final["ugpa"], "sat": valid_sat, "c_sat": valid_c_sat,
                  "sat_i": law_valid_dic_final["lsat"], "fya": law_valid_dic_final["zfya"],
                  "a": law_valid_dic_final["a"]}
    test_data = {"K": np.mean(test_K, axis=1), "gpa": test_gpa, "c_gpa": test_c_gpa,
                  "gpa_i": law_test_dic_final["ugpa"], "sat": test_sat, "c_sat": test_c_sat,
                  "sat_i": law_test_dic_final["lsat"], "fya": law_test_dic_final["zfya"],
                  "a": law_test_dic_final["a"]}
    np.savez("./saved_models/train_data_{}.npz".format(args.seed), **train_data)
    np.savez("./saved_models/valid_data_{}.npz".format(args.seed), **valid_data)
    np.savez("./saved_models/test_data_{}.npz".format(args.seed), **test_data)

if __name__ == "__main__":
    main()