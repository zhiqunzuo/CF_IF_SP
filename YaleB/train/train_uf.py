from loss import *
from util.utils import *
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from model.base import BestClf
from model.uf import UFModel
import torch.nn.functional as F

import os
import collections
# import wandb
# import pdb

def train_uf_yaleb(args, model, train_loader, test_loader, device, model_file_name):
    current_iter = 0
    tot_train_iters = (190 // args.batch_size) * args.epochs
    print(f"total train steps : {tot_train_iters}")

    # model / optimizer setup
    opt_model = torch.optim.Adam(model.parameters(), lr=args.lr)

    lrs = []
    if args.scheduler == 'lr':
        scheduler_model = LinearLR(opt_model, total_iters=args.epochs, last_epoch=-1, start_factor=1.0, end_factor=args.end_fac)
        schedulers = [scheduler_model]
        pass
    elif args.scheduler == 'one':
        scheduler_model = OneCycleLR(opt_model, max_lr=args.max_lr, steps_per_epoch=190 // args.batch_size, epochs=args.epochs, anneal_strategy='linear')
        schedulers = [scheduler_model]
    else:
        schedulers = []
    
    cf_lossf = nn.CrossEntropyLoss()

    best_pred_acc, best_to, patience = -1e7, -1e7, 0
    for epoch in range(1, args.epochs + 1):
        # ------------------------------------
        # Training
        # ------------------------------------
        model.train()
        ep_tot_loss, ep_pred_cor, ep_tot_num = 0.0, 0.0, 0.0

        for x, s, y in train_loader:
            current_iter += 1
            n = x.shape[0]

            # out : (zx, zs), (x_recon, s_recon), (mu_x, logvar_x, mu_s, logvar_s), y_pred
            out1 = model(x)
            y_pred = out1[1]
            loss = cf_lossf(out1[1], y)

            opt_model.zero_grad(set_to_none=True)

            loss.backward()

            # gradient clipping for vae backbone
            #if args.clip_val != 0:
            #    torch.nn.utils.clip_grad_norm_(model.vae_params(), max_norm=args.clip_val)
            opt_model.step()

            # update scheduler (iter wise)
            if args.scheduler == 'one':
                for sche in schedulers:
                    sche.step()

            # Train loss tracking
            ep_tot_num += n
    
            
            y_pred = torch.argmax(y_pred, dim=1)
            ep_pred_cor += (y_pred == y).sum().item()
            ep_tot_loss += loss.item() * n

        if args.scheduler == 'lr':
            for sche in schedulers:
                sche.step()
            lrs.append(opt_model.param_groups[0]['lr'])

        # ------------------------------------
        # validation
        # ------------------------------------
        model.eval()
        ep_tot_test_num, ep_pred_cor_test = 0, 0
        current_to = 0.0

        y_pred_raw = torch.tensor([], device=device)
        with torch.no_grad():
            for x, s, y in test_loader:
                n = x.shape[0]
                x = x.float(); y = y.long()

                out1 = model(x)

                loss = cf_lossf(out1[1], y)
                y_pred_te = torch.argmax(out1[1], dim=1)
                    
                ep_tot_test_num += n
                y_pred_raw = torch.cat((y_pred_raw, y_pred_te))
                ep_pred_cor_test += (y_pred_te == y).sum().item()

            
        # save farcon w.r.t epoch best trade off
        ep_pred_test_acc = ep_pred_cor_test / ep_tot_test_num
        if best_pred_acc < ep_pred_test_acc:
            best_pred_acc = ep_pred_test_acc
            torch.save(model.state_dict(), os.path.join(args.model_path, 'cf_' + model_file_name))
            patience = 0
        else:
            if args.early_stop:
                patience += 1
                if (patience % 10) == 0 :
                    print(f"----------------------------------- increase patience {patience}/{args.patience}")
                if patience > args.patience:
                    print(f"----------------------------------- early stopping")
                    break
            else:
                pass

        if (epoch % 10) == 0:
            print(f'\nEp: [{epoch}/{args.epochs}] (TRAIN) ---------------------\n'
                f'Loss: {ep_tot_loss / ep_tot_num:.3f}, L_pred_acc: {ep_pred_cor / ep_tot_num:.3f}')
            print(f'Ep: [{epoch}/{args.epochs}] (VALID)  ---------------------\n'
                f'L_pred_acc: {ep_pred_cor_test / ep_tot_test_num:.3f}')

    # log_dict['best_pred_test_acc'] = best_pred_acc

    # return bestclf

    # return best vs last model
    if args.last_epmod :
        return model
    else:
        model = UFModel(args, device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, 'cf_' + model_file_name), map_location=device))
        model.to(device)
        return model