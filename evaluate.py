import time
import datetime
from torch.nn import parameter
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import cv2
import utils.visualize as vis

from utils.cs_dataset import cs_data
from utils.cs_eval_object_detection_3d import main_eval
from cfg.settings import get_cfg_defaults
from models.dla.dlaseg import build_models
from utils.evaluation import save_results_as_json

import torch
from torch import optim
from torch.utils.data import dataloader

from torchinfo import summary

def eval(cfg):
    
    cs_dir = cfg.DATASETS.ROOT

    cs_eval_data = cs_data(cs_dir, cfg=cfg, split='val')
    cs_eval_loader = dataloader.DataLoader(cs_eval_data, batch_size=cfg.EVAL.BATCH_SIZE, num_workers=4, pin_memory=True)

    model = build_models(cfg=cfg)

    if cfg.TRAINING.USE_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    else: device = 'cpu'

    chkpt = cfg.EVAL.MODEL
    chkpt = torch.load(chkpt)
    pretrained_model = chkpt['model_state_dict']
    model.load_state_dict(pretrained_model)

    with torch.no_grad():
        model.eval()
        nb_v = len(cs_eval_loader)
        vbar = tqdm(enumerate(cs_eval_loader), total=nb_v)
        for i, batch in vbar:
            for key in batch:
                if key == 'label_idx': continue
                batch[key] = batch[key].to(device)
            input = batch['image']

            with torch.cuda.amp.autocast():
                output = model(input)
            
            save_results_as_json(output, batch, cfg)
    
if __name__ == '__main__':
    cfg = get_cfg_defaults()
    eval(cfg) 
    main_eval(cfg)
