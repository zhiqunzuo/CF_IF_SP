import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from utils import seed_everything

def result_report(model_name):
    mses = []
    tes = []
    s_accs = []
    models = []
    
    for seed in range(42, 47):
        with open("representation/{}_{}.pkl".format(model_name, seed), "rb") as f:
            data = pickle.load(f)
        
        model = LinearRegression()
        model.fit(np.concatenate([data["train_h"], data["valid_h"]], axis=0),
                  np.concatenate([data["train_y"], data["valid_y"]], axis=0))
        pred = model.predict(data["test_h"])
        mse = mean_squared_error(data["test_y"], pred)
        mses.append(mse)
        
        c_pred = model.predict(data["test_c_h"])
        te = np.mean(np.abs(pred - c_pred))
        tes.append(te)
        
        model = SVC()
        model.fit(np.concatenate([data["train_h"], data["valid_h"]], axis=0),
                  np.concatenate([data["train_a"], data["valid_a"]], axis=0))
        pred = model.predict(data["test_h"])
        s_acc = accuracy_score(data["test_a"], pred)
        s_accs.append(s_acc)
    
    return mses, tes, s_accs

def main():
    seed_everything(seed=42)
    mses, tes, s_accs = result_report(model_name="FarconVAE-t")
    print("FarconVAE-t")
    print("mse = {} + {}, te = {} + {}, s_acc = {} + {}".format(
        np.mean(mses), np.std(mses), np.mean(tes), np.std(tes), np.mean(s_accs), np.std(tes)
    ))
    
    mses, tes, s_accs = result_report(model_name="FarconVAE-G")
    print("FarconVAE-G")
    print("mse = {} + {}, te = {} + {}, s_acc = {} + {}".format(
        np.mean(mses), np.std(mses), np.mean(tes), np.std(tes), np.mean(s_accs), np.std(tes)
    ))
    
    mses, tes, s_accs = result_report(model_name="MaxEnt-ARL")
    print("MaxEnt-ARL")
    print("mse = {} + {}, te = {} + {}, s_acc = {} + {}".format(
        np.mean(mses), np.std(mses), np.mean(tes), np.std(tes), np.mean(s_accs), np.std(tes)
    ))
    
    mses, tes, s_accs = result_report(model_name="CI")
    print("CI")
    print("mse = {} + {}, te = {} + {}, s_acc = {} + {}".format(
        np.mean(mses), np.std(mses), np.mean(tes), np.std(tes), np.mean(s_accs), np.std(tes)
    ))
    
    mses, tes, s_accs = result_report(model_name="UF")
    print("UF")
    print("mse = {} + {}, te = {} + {}, s_acc = {} + {}".format(
        np.mean(mses), np.std(mses), np.mean(tes), np.std(tes), np.mean(s_accs), np.std(tes)
    ))
    
    mses, tes, s_accs = result_report(model_name="ours")
    print("ours")
    print("mse = {} + {}, te = {} + {}, s_acc = {} + {}".format(
        np.mean(mses), np.std(mses), np.mean(tes), np.std(tes), np.mean(s_accs), np.std(tes)
    ))

if __name__ == "__main__":
    main()