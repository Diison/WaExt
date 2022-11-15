# -*- coding: utf-8 -*-

import torch

def process_acc(w_pred, weight, idx, size, mode, th):

    """
    mode = 0: 直接返回 w_pred
         = 1: 返回 diff = abs(weight-w_pred)
         = 2: 返回 acc = abs(weight-w_pred) / weight
         = 3: 返回 exp(abs(weight-w_pred) / weight)
         = 4: 返回 exp[-(abs(weight-w_pred) / weight)]
    """
    if idx is None:
        diff=abs(weight.to(device=w_pred.device)-w_pred)
        if mode==0:
            return w_pred
        elif mode==1:
            return diff
        elif mode==2:
            return diff/weight
        elif mode==3:
            return torch.exp(diff/weight)
        elif mode==4:
            return torch.exp(-diff/weight)
    else:
        x = torch.tensor(idx[:,0])
        y = torch.tensor(idx[:,1])
        diff=abs(weight-w_pred).to(device=w_pred.device)
        all_acc = torch.zeros(size[0], size[1]).to(device=w_pred.device)
        
        if mode==0:
            all_acc[x, y] = w_pred
            return all_acc
        elif mode==1:
            all_acc[x, y] = diff
            return all_acc
        elif mode==2:
            all_acc[x, y] = diff/weight
            return all_acc
        elif mode==3:
            all_acc[x, y] = torch.exp(diff/weight-th)
            return all_acc
        elif mode==4:
            all_acc[x, y] = torch.exp(-(diff/weight-th))
            return all_acc
