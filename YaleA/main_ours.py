from random import choice
from train.train_cf import *
from model.cf import *
from eval import *

import torch
import datetime
import time


def main(args, device):
    start_time = time.time()
    print(f'using {device}')
    seed_everything(args.seed)
    RUN_ID = str(datetime.datetime.now()).replace(' ','_').replace('-','_').replace('.',':').replace(':','_')

    # saving file names define
    if args.save_name == 'default':
        model_file_name = RUN_ID + '_' + args.model_name + '.pth'
        z_vis_name = RUN_ID + '_' + args.model_name + '_vis' + '.png'
    else:
        model_file_name = args.save_name + '.pth'
        z_vis_name = args.save_name + '_vis' + '.png'
    args.down_save_name = model_file_name

    
    data = CFYaleBDataLoader(args.data_path, args=args, device=device)
    data.load()
    train_loader = torch.utils.data.DataLoader(data.trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data.testset, batch_size=args.batch_size_te, shuffle=False, num_workers=0)

    model = CFModel(args, device)
    model.to(device)
    
    # train model
    model = train_cf_yaleb(args, model, train_loader, test_loader, device, model_file_name)

    return model, data

def evaluation(args, device, eval_model, model=None, is_e2e=True, trainset=None, testset=None):
    seed_everything(args.seed)
    print(f'using {device}')

    # Validate
    if is_e2e:
        print('do End-to-End Experiment phase')
        y_pred, y_acc, s_acc, s_pred, cf = z_evaluator(args, model, None, device, eval_model, 'cf', trainset, testset)
    else:
        model = FarconVAE(args, device)
        clf_xy = BestClf(args.n_features, 1, args.hidden_units, args)
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_file), map_location=device))
        clf_xy.load_state_dict(torch.load(os.path.join(args.model_path, 'clf_'+args.model_file), map_location=device))
        model, clf_xy = model.to(device), clf_xy.to(device)

        # predict Y, S from learned representation using eval_model(LR or MLP) (s is always evaluated with MLP according to previous works)
        y_pred, y_acc, s_acc, s_pred = z_evaluator(args, model, clf_xy, device, eval_model, 'ours', trainset, testset)

    # Final reporting
    if args.data_name != 'yaleb':
        dp, _, _, eodd, gap = metric_scores(os.path.join(args.data_path, args.test_file_name), y_pred)
        print('----------------------------------------------------------------')
        print(f'DP: {dp:.4f} | EO: {eodd:.4f} | GAP: {gap:.4f} | yAcc: {y_acc:.4f} | sAcc: {s_acc:.4f}')
        print('----------------------------------------------------------------')
        #performance_log = {f'DP': dp, f'EO':eodd, f'GAP': gap, f'y_acc': y_acc, f's_acc': s_acc}
        #wandb.log(performance_log)
    else:
        keys = np.unique(np.array(s_pred), return_counts=True)[0]
        values = np.unique(np.array(s_pred), return_counts=True)[1]/s_pred.shape[0]
        s_pred_log = {'pred_'+str(i):0 for i in range(5)}
        for i in range(len(keys)):
            s_pred_log['pred_'+str(keys[i])] = values[i]
        print('----------------------------------------------------------------')
        print(f'yAcc: {y_acc:.4f} | sAcc: {s_acc:.4f} | cf: {cf:.4f}')
        print('----------------------------------------------------------------')
        # performance_log = {f'y_acc': y_acc, f's_acc': s_acc}
        # wandb.log(s_pred_log)
        # wandb.log(performance_log)
    return y_acc, s_acc, cf

# -----------------------
# end-to-end [train & evaluation]
# -----------------------
def e2e(args, device):
    seed_everything(args.seed)
    start_time = time.time()

    y_accs = []; s_accs = []; cfs = []
    for it_seed in range(730, 730+args.n_seed):
        if args.n_seed != 1:
            args.seed = it_seed

        model, data = main(args, device)
        
        eval_model = args.eval_model if args.eval_model != 'disc' else args.disc

        y_acc, s_acc, cf = evaluation(args, device, eval_model, model, is_e2e=True, trainset=data, testset=data)
        
        y_accs.append(y_acc); s_accs.append(s_acc), cfs.append(cf)
    if args.n_seed != 1:
        y_mean, y_std = np.mean(y_accs), np.std(y_accs)
        s_mean, s_std = np.mean(s_accs), np.std(s_accs)
        cf_mean, cf_std = np.mean(cfs), np.std(cfs)
        print('\n-----------------------------------------------------------')
        print(f"{args.n_seed} runs y acc mean: {y_mean}, std: {y_std},\n       s acc means: {s_mean}, std: {s_std},\n fc means: {cf_mean}, std: {cf_std}")
        print('---------------------------------------------------------')
    print('\n-----------------------------------------------------------')
    print(f"Time for Train & Eval per 1 program:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('---------------------------------------------------------')

if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser()
    # ------------------- run config
    parser.add_argument("--seed", type=int, default=730, help="one manual random seed")
    parser.add_argument("--n_seed", type=int, default=1, help="number of runs")
    parser.add_argument("--run_mode", type=str, default='e2e', choices=["train", "eval", "e2e"])

    # ------------------- flag & name
    parser.add_argument("--data_name", type=str, default='yaleb', help="Dataset name")
    parser.add_argument("--model_name", type=str, default='cf', help="Model ID")
    parser.add_argument("--save_name", type=str, default='default', help="specific string or system time at run start(by default)")
    parser.add_argument("--env_flag", type=str, default='nn', help="env noise flag [s,y]")
    parser.add_argument("--env_eps", type=float, default=0.15, help="env noise proportion")
    parser.add_argument("--tr_ratio", type=float, default=1.0, help="train set use rate")

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

    args = parser.parse_args()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------------- specific argument setup for each dataset
    args.drop_p = 0.3
    args.neg_slop = 0.1
    args.clf_layers = 2
    args.cont_xs = 1
    if args.env_flag != 'nn':
        args.env_eps_s_tr = args.env_eps
        args.env_eps_y_tr = args.env_eps

    if args.data_name == "yaleb":
        args.y_dim = 38
        args.s_dim = 5
        args.n_features = 504
        args.latent_dim = 100
        args.hidden_units = 100
        args.encoder = 'lr'
        args.batch_size = 190
        args.batch_size_te = 380
        args.epochs = 2000
        args.fade_in = 1  # annealing flag for DCD, SR loss
        args.beta_anneal = 1  # annealing flag for KLD loss
        args.clf_act = 'prelu'
        args.clf_seq = 'fad'
        args.clf_hidden_units = 75
        args.connection = 2
        args.enc_act = 'prelu' if args.kernel == 't' else 'gelu'
        args.dec_act = 'prelu' if args.kernel == 't' else 'gelu'
        args.enc_seq = 'fba' if args.kernel == 't' else 'fa'
        args.dec_seq = 'f'
        args.pred_act = 'leaky'
        args.pred_seq = 'fba'
        args.model_path = './model_yaleb'
        args.data_path = './data/yaleb/'
        args.result_path = './result_yaleb'
        args.clf_path = './bestclf/bestclf_yaleb.pth'
    args.patience = int(args.epochs * 0.10)
    
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.isdir(args.result_path):
        os.mkdir(args.result_path)
        
    if args.run_mode == 'train':
        main(args, device)
    elif args.run_mode == 'eval':
        evaluation(args, device, eval_model='lr', is_e2e=False)
    else:
        e2e(args, device)