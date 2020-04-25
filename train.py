# -*- coding: utf-8 -*-
from __future__ import print_function
from utils.metrics import *
from utils.helper import *
from utils.Regularization import Regularization
import argparse
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
import warnings
import pickle
import torch


DATAPATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 128, help='batchsize')
parser.add_argument('--delta', type=float, default=0.5, help='delta')
parser.add_argument('--epsilon', type=float,default=1e-7)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float,default=0.999)
parser.add_argument('--lr', type=float,default=1e-3)
parser.add_argument('--target_year', type=str, default='2018')
parser.add_argument('--num_epochs',type=int, default=300)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='train', help='log_path')
parser.add_argument('--model_path',default='./model_save', help='manual seed')
parser.add_argument('--save', type=str, default='./experiment')
parser.add_argument('--model', type=str, default='HGMLT',choices=["LSTM","Bi_GRU", "Bi_GRU_ATT","HGMLT"])
parser.add_argument('--CACHEDATA', type=bool, default = True)
parser.add_argument('--T', type=int, default=24)
parser.add_argument('--len_test', type=int, default=24*7*13)    #13weeks is the best
parser.add_argument('--ext', type=bool, default=True)
parser.add_argument('--harved_epoch', type=int, default=30)
parser.add_argument('--dataset', type=str, default='PDP',choices=["DGS","PDS","PDP"],help='which dataset to use')
parser.add_argument('--sample_interval', type=int, default=60,help='interval between validation')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

warnings.filterwarnings('ignore')
save_path = 'saved_model/{}/{}-{}'.format(opt.dataset,
                                             opt.model,
                                             opt.ext)
os.makedirs(save_path, exist_ok=True)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:",device)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

iter = 0
rmses = [np.inf]
maes = [np.inf]

# Load Data
preprocess_file = DATAPATH+ f'/data/{opt.dataset}_{opt.target_year}_data.pkl'
print(preprocess_file)
# DGS_data = {"data": DGS_DG_norm, "LL_data": DGS_flow_norm, "temp_prices": TP, "date": DGS_DG_T,
#             "E_min": mmn._min, "E_max": mmn._max, "LL_min": ll_mmn._min, "LL_max": ll_mmn._max}
fpkl = open(preprocess_file, 'rb')
data = pickle.load(fpkl)
fpkl.close()
DGS_DG_norm = data['data']
DGS_flow_norm = data['LL_data']
DGS_DG_T = data['date']
mmn = data['E_mmn']
LL_mmn = data['LL_mmn']
# temperature_prices = data['temp_prices']
print(f'data length:{DGS_DG_norm.shape,DGS_flow_norm.shape,DGS_DG_T.shape}')


# model
print("feature layer:", opt.model)
if opt.model == 'LSTM':
    from model.baselines.LSTM import LSTM
    model = LSTM()
elif opt.model == 'Bi_GRU':
    from model.baselines.Bi_GRU import Bi_GRU
    model = Bi_GRU()
elif opt.model == 'Bi_GRU_ATT':
    from model.baselines.Bi_GRU_ATT import Bi_GRU_ATT
    model = Bi_GRU_ATT()
elif opt.model == 'HGMLT':
    from model.baselines.HGMLT_V5 import HGMLT
    model = HGMLT()
else:
    print("No model selected")
    exit(0)

loss_fn = nn.MSELoss()
model.apply(init_network_weights)



torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e10)
if cuda:
    model.cuda()
    loss_fn.cuda()
lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
print_model_parm_nums(model, opt.model)

if __name__ == '__main__':
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    train_batch_generator,test_batch_generator,external_dim = get_data_generator(opt,DGS_DG_norm,DGS_flow_norm,DGS_DG_T)
    s_time = datetime.now()

    #train
    for epoch in range(opt.num_epochs):
            ep_time = datetime.now()
            beta = min([(epoch * 1.) / max([100, 1.]), 1.])
            for i, (x_seq,y_batch,flow_x,flow_y,T) in enumerate(train_batch_generator):
                model.train()
                optimizer.zero_grad()
                """LSTM,Bi-GRU,Bi-GRU-ATT"""
                if opt.model == 'HGMLT' or opt.model == 'Bi_GRU' or opt.model == 'Bi_GRU_ATT':
                    pred_y,pred_flow_y,gaussian = model(x_seq,flow_x,flow_y,T)
                    gaussian_likelihood = gaussian.log_prob(y_batch)
                    #print(pred_y.shape,y_batch.shape,pred_flow_y.shape,flow_y.shape)
                    loss = 0.95*loss_fn(pred_y, y_batch) + 0.05*loss_fn(pred_flow_y,flow_y) - torch.mean(gaussian_likelihood,0)

                """GRU-VAE"""
                if opt.model == 'GRU_VAE' or opt.model == 'decayGRU_VAE':
                    pred_y, gaussian, kl_divergence,res_x = model(x_seq,flow_x,flow_y,T)
                    res_loss = loss_fn(torch.cat([x_seq, flow_x], dim=2), res_x)
                    gaussian_likelihood = gaussian.log_prob(y_batch)
                    gaussian_likelihood = torch.mean(gaussian_likelihood)
                    kl_divergence = torch.mean(kl_divergence)
                    ELBO = gaussian_likelihood - beta  * kl_divergence
                    loss =  - ELBO +res_loss
                    #print('loss:',format(loss,'.5f'),'KL:',format(kl_divergence,'.8f'),beta)

                # if weight_decay > 0:
                #     loss = loss + reg_loss(model)

                loss.backward()
                optimizer.step()
                iter += 1
            #test
            total_mse = 0
            model.eval()
            valid_time = datetime.now()
            with torch.no_grad():
                for i, (x_seq,y_batch,flow_x,flow_y,T) in enumerate(test_batch_generator):
                    if opt.model == 'HGMLT' or opt.model == 'Bi_GRU' or opt.model == 'Bi_GRU_ATT':
                        x_res,pred_flow_y,_ = model(x_seq, flow_x,flow_y,T)
                    else:
                        #GRU-VAE
                        x_res, _, _,_ = model(x_seq,flow_x,flow_y,T)
                    x_res = mmn.inverse_transform(x_res.cpu().detach().numpy())
                    y_batch = mmn.inverse_transform(y_batch.cpu().detach().numpy())
                    total_mse += get_MSE(x_res, y_batch) * len(x_seq)            #16
            rmse = np.sqrt(total_mse / len(test_batch_generator.dataset))  #3360
            if rmse < np.min(rmses):
                print("iter\t{}\tRMSE\t{:.6f}".format(iter, rmse))
                torch.save(model.state_dict(), '{}/{}-V5-2018.pt'.format(save_path,opt.model))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tRMSE\t{:.6f}\tmodel\t{}\n".format(epoch,iter,rmse,opt.model))
                f.close()
            rmses.append(rmse)

            # halve the learning rate
            if epoch % opt.harved_epoch == 0 and epoch != 0:
                lr /= 2
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("half the learning rate!\n")
                f.close()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))

    # #save model
    # torch.save(model.state_dict(), '{}/{}.pt'.format(save_path, opt.model))
    # #test
    # total_mse, total_mae, total_mape = 0, 0, 0
    # model.eval()
    # valid_time = datetime.now()
    # with torch.no_grad():
    #     for i, (x_seq, y_batch, flow_x, flow_y, T, TP) in enumerate(test_batch_generator):
    #         if opt.model == 'LSTM' or opt.model == 'Bi_GRU' or opt.model == 'Bi_GRU_ATT':
    #             pred_y = model(x_seq, flow_x)
    #
    #         if opt.model == 'GRU_VAE' or opt.model == 'PlanarVAE' or opt.model == 'LatentODE' :
    #             pred_y,_,_ = model(x_seq,flow_x)
    #
    #         if opt.model == 'DeepHydro':
    #             pred_y,_,_,_ = model(x_seq,flow_x,flow_y,T,TP)
    #
    #         pred_y = mmn.inverse_transform(pred_y.cpu().detach().numpy())
    #         y_batch = mmn.inverse_transform(y_batch.cpu().detach().numpy())
    #         total_mse += get_MSE(pred_y, y_batch) * len(x_seq)
    #         total_mae += get_MAE(pred_y, y_batch) * len(x_seq)
    #         total_mape += get_MAPE(pred_y, y_batch) * len(x_seq)
    # rmse = np.sqrt(total_mse / len(test_batch_generator.dataset))
    # mae = total_mae / len(test_batch_generator.dataset)
    # mape = total_mape / len(test_batch_generator.dataset)
    # print('Test\tRMSE\t{:.6f}\tMAE\t{:.6f}\tMAPE\t{:.6f}'.format(rmse, mae, mape))