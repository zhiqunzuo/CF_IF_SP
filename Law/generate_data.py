import numpy as np
import os
import pandas as pd
import random

from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = str(42)


parameter_dic = {
    "ugpa0": 0.1,
    "eta_u_ugpa": 0.3,
    "eta_a_ugpa": np.array([0.2, 0.1, 0.3, 0.4, 0.7, 0.1, 0.2, 0.4, 0.5, 0.2]),
    "lsat0": 0.3,
    "eta_u_lsat": 0.6,
    "eta_a_lsat": np.array([0.4, 0.3, 0.1, 0.5, 0.6, 0.4, 0.3, 0.7, 0.8, 0.6]),
    "eta_u_zfya": 0.5,
    "eta_a_zfya": np.array([0.5, 0.2, 0.6, 0.7, 0.2, 0.3, 0.1, 0.6, 0.8, 0.4])
}

"""
mean = np.zeros(10)
std = np.identity(10)

ugpa0 = np.random.normal(0, 1)
eta_u_ugpa = np.random.normal(0, 1)
eta_a_ugpa = np.random.multivariate_normal(mean, std)
lsat0 = np.random.normal(0, 1)
eta_u_lsat = np.random.normal(0, 1)
eta_a_lsat = np.random.multivariate_normal(mean, std)
eta_u_zfya = np.random.normal(0, 1)
eta_a_zfya = np.random.multivariate_normal(mean, std)

parameter_dic = {
    "ugpa0": ugpa0, "eta_u_ugpa": eta_u_ugpa, "eta_a_ugpa": eta_a_ugpa,
    "lsat0": lsat0, "eta_u_lsat": eta_u_lsat, "eta_a_lsat": eta_a_lsat,
    "eta_u_zfya": eta_u_zfya, "eta_a_zfya": eta_a_zfya
}
print(parameter_dic)
"""

race_names = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other",
              "Puertorican", "White"]
gender_names = [1, 2]

def sample_u():
    K = np.random.uniform(low=0, high=1, size=30000)
    return K

def sample_a():
    race = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=30000)
    sex = np.random.choice([0, 1], size=30000)
    return race, sex

def one_hot_a(race, sex):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_race = encoder.fit_transform(race.reshape(-1, 1))
    encoded_sex = encoder.fit_transform(sex.reshape(-1, 1))
    a = np.concatenate([encoded_race, encoded_sex], axis=1)
    return a

def get_gpa(K, a, parameters):
    gpa = parameters["ugpa0"] + parameters["eta_u_ugpa"] * K + np.dot(a, parameters["eta_a_ugpa"])
    c_a = np.concatenate([a[:, :8], 1 - a[:, 8:]], axis=1)
    c_gpa = parameters["ugpa0"] + parameters["eta_u_ugpa"] * K + np.dot(c_a, parameters["eta_a_ugpa"])
    return gpa, c_gpa

def get_sat(K, a, parameters):
    sat = parameters["lsat0"] + parameters["eta_u_lsat"] * K + np.dot(a, parameters["eta_a_lsat"])
    c_a = np.concatenate([a[:, :8], 1 - a[:, 8:]], axis=1)
    c_sat = parameters["lsat0"] + parameters["eta_u_lsat"] * K + np.dot(c_a, parameters["eta_a_lsat"])
    return sat, c_sat

def get_fya(K, a, parameters):
    fya = parameters["eta_u_zfya"] * K + np.dot(a, parameters["eta_a_zfya"])
    return fya

def main():
    K = sample_u()
    race, sex = sample_a()
    a = one_hot_a(race, sex)
    gpa, c_gpa = get_gpa(K, a, parameter_dic)
    sat, c_sat = get_sat(K, a, parameter_dic)
    fya = get_fya(K, a, parameter_dic)
    data = {
        "K": K,
        "race": [race_names[race_num] for race_num in race],
        "sex": [gender_names[sex_num] for sex_num in sex],
        "LSAT": sat,
        "c_LSAT": c_sat,
        "UGPA": gpa,
        "c_UGPA": c_gpa,
        "ZFYA": fya
    }
    """for i, key in enumerate(data.keys()):
        if i <= 1:
            print("{}, shape = {}".format(key, len(data[key])))
        else:
            print("{}, shape = {}".format(key, data[key].shape))"""
    data = pd.DataFrame(data)
    #data.to_csv("generated_data_2.csv", index=True)
    data.to_csv("generated_data.csv", index=False)

if __name__ == "__main__":
    main()