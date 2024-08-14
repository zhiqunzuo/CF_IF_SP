import argparse
import datetime
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch

from dataset import get_xsy_loaders
from model.maxent import MaxEnt
from train.train_maxent import train_maxent
from utils import seed_everything

def main(args, device):
    start_time = time.time()
    seed_everything(args.seed)
    RUN_ID = str(datetime.datetime.now()).replace(' ','_').replace('-','_').replace('.',':').replace(':','_')
    
    model_file_name = RUN_ID + "_" + args.model_name + ".pth"
    
    train_loader, valid_loader = get_xsy_loaders(os.path.join("data", args.train_file_name),
                                                 os.path.join("data", args.valid_file_name),
                                                 args.sensitive, args.batch_size_te, args)
    __, test_loader = get_xsy_loaders(os.path.join("data", args.train_file_name),
                                      os.path.join("data", args.valid_file_name),
                                      args.sensitive, args.batch_size_te, args)
    
    model = MaxEnt(args, device)
    model.to(device)
    model = train_maxent(args, train_loader, valid_loader, model, device, model_file_name)
    
    return model, train_loader.dataset, valid_loader.dataset, test_loader.dataset

def process_dataset(path):
    data = pd.read_csv(path)
    data = pd.get_dummies(data, columns=["race"], prefix="", prefix_sep="")
    race_names = ["Amerindian", "Asian", "Black", "Hispanic", "Mexican", "Other",
                  "Puertorican", "White"]
    data[race_names] = data[race_names].astype(int)
    
    dataset = {}
    dataset["s"] = np.array(data["sex"]) - 1
    dataset["cs"] = 1 - dataset["s"]
    dataset["X"] = np.concatenate([np.array(data["LSAT"]).reshape(-1, 1),
                                   np.array(data["UGPA"]).reshape(-1, 1),
                                   np.array(data[race_names])], axis=1)
    dataset["cX"] = np.concatenate([np.array(data["c_LSAT"]).reshape(-1, 1),
                                    np.array(data["c_UGPA"]).reshape(-1, 1),
                                    np.array(data[race_names])], axis=1)
    dataset["y"] = np.array(data["ZFYA"])
    return dataset

def encode(dataset, model):
    X = dataset["X"]
    cX = dataset["cX"]
    s = dataset["s"]
    cs = dataset["cs"]
    
    X_tensor = torch.FloatTensor(X).cuda()
    cX_tensor = torch.FloatTensor(cX).cuda()
    
    model.eval()
    
    with torch.no_grad():
        zx, __, __ = model(X_tensor)
        czx, __, _ = model(cX_tensor)
    
    return np.array(zx.cpu()), np.array(czx.cpu())

def save_representation(args, model):
    train_set = process_dataset(os.path.join("data", "generated_train_{}_counter.csv".format(args.seed)))
    valid_set = process_dataset(os.path.join("data", "generated_valid_{}_counter.csv".format(args.seed)))
    test_set = process_dataset(os.path.join("data", "generated_test_{}_counter.csv".format(args.seed)))
    
    train_zx, train_czx = encode(train_set, model)
    valid_zx, valid_czx = encode(valid_set, model)
    test_zx, test_czx = encode(test_set, model)
    
    data = {"train_h": train_zx, "train_c_h": train_czx, "train_a": train_set["s"], "train_y": train_set["y"],
            "valid_h": valid_zx, "valid_c_h": valid_czx, "valid_a": valid_set["s"], "valid_y": valid_set["y"],
            "test_h": test_zx, "test_c_h": test_czx, "test_a": test_set["s"], "test_y": test_set["y"]}
    
    with open("representation/{}_{}.pkl".format(args.model_name, args.seed), "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # model name
    parser.add_argument("--model_name", type=str, help="model used in the experimemt")
    
    # ------------------- train config
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--scheduler", type=str, default='None', choices=['None', 'one', 'lr'], help="learning rate scheduler")
    parser.add_argument("--max_lr", type=float, default=3e-2, help="anneal max lr")
    parser.add_argument("--end_fac", type=float, default=0.001, help="linear decay end factor")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip_val", type=float, default=2.0, help="gradient clipping value for VAE backbone when using DCD loss")
    parser.add_argument("--early_stop", type=int, default=0, choices=[1, 0])
    parser.add_argument("--last_epmod", type=int, default=0, choices=[1, 0], help='use last epoch model or best model at the end of 1 stage train')
    parser.add_argument("--last_epmod_eval", type=int, default=1, choices=[1, 0], help='use last epoch model or best model for evaluation')
    parser.add_argument("--eval_model", type=str, default='lr', choices=['mlp', 'lr', 'disc'], help='representation quality evaluation model')

    # ------------------- loss config
    parser.add_argument("--kernel", type=str, default='g', choices=['t', 'g'], help="DCD loss kernel type")
    parser.add_argument("--alpha", type=float, default=1.0, help="DCD loss weight")
    parser.add_argument("--beta", type=float, default=0.2, help="KLD prior regularization weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="swap recon loss weight")
    
    # ------------------ MaxEnt config
    parser.add_argument("--disc_y", type=str, default="mlp")
    parser.add_argument("--kld_agg", type=str, default="sum")
    
    args = parser.parse_args()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    args.drop_p = 0.3
    args.neg_slop = 0.1
    args.clf_layers = 2
    args.cont_xs = 1
    
    # data related settings
    args.y_dim = 1
    args.s_dim = 1
    args.n_features = 10
    args.latent_dim = 5
    args.hidden_units = 8
    args.sensitive = "sex"
    args.target = "ZFYA"
    args.encoder = "mlp"
    args.batch_size = 17342
    args.batch_size_te = 2179
    args.epochs = 300
    args.fade_in = 1
    args.beta_anneal = 0
    args.clf_act = "leaky"
    args.clf_seq = "fad"
    args.clf_hidden_units = 8
    args.connection = 0
    args.enc_act = "gelu"
    args.dec_act = "gelu"
    args.enc_seq = "fa"
    args.dec_seq = "fa"
    args.pred_act = "leaky"
    args.pred_seq = "fba" if args.kernel == "t" else "fad"
    args.clf_path = "no"
    args.save_criterion = "yloss"
    
    for seed in range(42, 47):
        args.seed = seed
        args.model_path = "./model_generated_{}".format(seed)
        args.result_path = "./result_generated_{}".format(seed)
        
        args.train_file_name = "generated_train_{}.csv".format(seed)
        args.valid_file_name = "generated_valid_{}.csv".format(seed)
        args.test_file_name = "generated_test_{}.csv".format(seed)
        
        if not os.path.isdir(args.model_path):
            os.mkdir(args.model_path)
        if not os.path.isdir(args.result_path):
            os.mkdir(args.result_path)
        
        model, __, __, __ = main(args, device)
        save_representation(args, model)
        