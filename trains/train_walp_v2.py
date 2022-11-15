import argparse
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../models")))

from wtrans_e import WTransE
from wtrans_h import WTransH
from wdistmult import WDistMult
from wcomplex import WComplEx
from slcwa_eval_walp import SLCWATrainingLoopWaLPEval
from rank_based_evaluator import WeightAwareRankBasedEvaluator
from classification_evaluator import SampledWeightAwareClassificationEvaluator, WeightAwareClassificationEvaluator
from trans_e import TransE
from trans_h import TransH
from distmult import DistMult
from complex import ComplEx
from my_complex import MyComplEx
from my_distmult import MyDistMult
from my_trans_e import MyTransE
from my_trans_h import MyTransH

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../datasets")))
print(sys.path)
from nl27k import load_nl27k
from cn15k import load_cn15k
from ppi5k import load_ppi5k


from pykeen.evaluation import RankBasedEvaluator

import torch
torch.manual_seed(1024)
from torch.optim import Adam
import numpy as np

# ------------------
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used
# ------------------


# def weighting_func(weight, base=np.e):
#     return np.power(base, weight)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default="WTransE")
    parser.add_argument('--dataset', default="CN15K")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--save_dir', default="_")
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--train_batch', type=int, default=256)
    parser.add_argument('--eval_batch', type=int, default=256)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--n_weight', type=float, default=0.5)
    parser.add_argument('--th', type=float, default=0.8)
    parser.add_argument('--dth', type=float, default=0)
    parser.add_argument('--dpct', type=float, default=1)
    parser.add_argument('--embedding_dim', type=int, default=50)
    parser.add_argument('--base', type=float, default=np.e)
    parser.add_argument('--base_mode', default="static")


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args=parse_args()
    model_name=args.model
    dataset_name=args.dataset
    device=args.device
    save_dir=args.save_dir
    epoch=args.epoch
    train_batch=args.train_batch
    eval_batch=args.eval_batch
    n_weight=args.n_weight
    th=args.th
    dth=args.dth
    dpct=args.dpct
    embedding_dim=args.embedding_dim
    step=args.step
    base=args.base
    base_mode=args.base_mode
    
    if save_dir=="_":
        save_dir=os.path.abspath(os.path.join(os.getcwd(), "../results"))

    print("args: ", args)

    def weighting_func(weight, base=args.base):
        return np.power(base, weight)
        
    if dataset_name.lower()=="cn15k":
        dataset=load_cn15k(th=dth, pct=dpct)
    elif dataset_name.lower()=="nl27k": 
        dataset=load_nl27k(th=dth, pct=dpct)
    elif dataset_name.lower()=="ppi5k": 
        dataset=load_ppi5k(th=dth, pct=dpct)

    if n_weight == -100.0:
        n_weight = np.mean(list(dataset.train_quad.values()))
        print("\n THE WEIGHT OF THE NEGs IS: ", n_weight, "\n")
    elif n_weight == -200.0:
        n_weight = 1.0 - np.mean(list(dataset.train_quad.values()))
        print("\n THE 1 - WEIGHT OF THE NEGs IS: ", n_weight, "\n")

    if model_name.lower().startswith("w"):
        if base_mode == "dynamic":
            save_location = save_dir+"/"+str(th)+"_dpct_"+str(dpct)+"_step_"+str(step)+"/"+str(base_mode)+'/'+"nw_"+str(n_weight)+'/'+'/'+model_name+'/'+dataset_name 
        elif base_mode == "static":
            save_location = save_dir+"/"+str(th)+"_dpct_"+str(dpct)+"_step_"+str(step)+"/base_"+str(base)+'/'+"nw_"+str(n_weight)+'/'+'/'+model_name+'/'+dataset_name 
    elif model_name.lower().startswith("my"):
        save_location = save_dir+"/baseline_my/"+"_dpct_"+str(dpct)+"_step_"+str(step)+"_base_"+str(base)+'/'+model_name+'/'+dataset_name 
    else:
        save_location = save_dir+"/baseline/"+"_dpct_"+str(dpct)+"_step_"+str(step)+"_base_"+str(base)+'/'+model_name+'/'+dataset_name 

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    if model_name.lower()=="wcomplex":
        model = WComplEx(random_seed=1024, embedding_dim=embedding_dim, wfunc=weighting_func, triples_factory=dataset.training, quads=dataset.quads, n_weight=n_weight, device=device, base_mode=base_mode)
    elif model_name.lower()=="complex":
        model = ComplEx(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)
    elif model_name.lower()=="wtranse":
        model = WTransE(random_seed=1024, embedding_dim=embedding_dim, wfunc=weighting_func, triples_factory=dataset.training, quads=dataset.quads, n_weight=n_weight, device=device, base_mode=base_mode)
    elif model_name.lower()=="transe":
        model = TransE(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)
    elif model_name.lower()=="wtransh":
        model = WTransH(random_seed=1024, embedding_dim=embedding_dim, wfunc=weighting_func, triples_factory=dataset.training, quads=dataset.quads, n_weight=n_weight, device=device, base_mode=base_mode)
    elif model_name.lower()=="transh":
        model = TransH(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)
    elif model_name.lower()=="wdistmult":
        model = WDistMult(random_seed=1024, embedding_dim=embedding_dim, wfunc=weighting_func, triples_factory=dataset.training, quads=dataset.quads, n_weight=n_weight, device=device, base_mode=base_mode)
    elif model_name.lower()=="distmult":
        model = DistMult(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)

    elif model_name.lower()=="mycomplex":
        model = MyComplEx(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)
    elif model_name.lower()=="mydistmult":
        model = MyDistMult(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)
    elif model_name.lower()=="mytranse":
        model = MyTransE(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)
    elif model_name.lower()=="mytransh":
        model = MyTransH(random_seed=1024, embedding_dim=embedding_dim, triples_factory=dataset.training)

    model.to(torch.device(device))
    
    print("Class Location: ", os.path.abspath(sys.modules[model.__class__.__module__].__file__))
    with open(save_location+"/"+"class_loc.txt", "w") as classf:
        classf.write(os.path.abspath(sys.modules[model.__class__.__module__].__file__))

    lp_evaluator = RankBasedEvaluator(filtered=True)
    walp_evaluator = WeightAwareRankBasedEvaluator(filtered=True, quads=dataset.quads, weighting_func = weighting_func)
    swtc_evals = {}
    swtc_evaluator_100 = SampledWeightAwareClassificationEvaluator(evaluation_factory=dataset._testing, additional_filter_triples=[dataset._training.mapped_triples, dataset._validation.mapped_triples], num_negatives=100, quads=dataset.quads, weighting_func = weighting_func)
    swtc_evals['100'] = swtc_evaluator_100

    optimizer = Adam(params=model.get_grad_params())
    
    if model_name.startswith("W"):
        print("save_location: ", save_location)
        training_loop = SLCWATrainingLoopWaLPEval(model=model, 
                                              optimizer=optimizer, 
                                              lp_evaluator = lp_evaluator,
                                              walp_evaluator = walp_evaluator,
                                              save_location = save_location,
                                              swtc_evals = swtc_evals,
                                              dataset = dataset,
                                              step = step,
                                              eval_batch = eval_batch) 
        training_loop.train(triples_factory=dataset.training, num_epochs=epoch, batch_size=train_batch)

    else:
        print("save_location: ", save_location)
        training_loop = SLCWATrainingLoopWaLPEval(model=model, 
                                              optimizer=optimizer, 
                                              lp_evaluator = lp_evaluator,
                                              walp_evaluator = walp_evaluator,
                                              save_location = save_location,
                                              swtc_evals = swtc_evals,
                                              dataset = dataset,
                                              step = step,
                                              eval_batch = eval_batch) 
        training_loop.train(triples_factory=dataset.training, num_epochs=epoch, batch_size=train_batch)
