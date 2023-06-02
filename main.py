import yaml
import pprint
import os
import argparse
from solver import Solver
from data_loader import get_loader, decision_model_get_loader
from torch.backends import cudnn
import torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import copy
from torch import topk
from torch.nn import functional as F
import cv2
from Decision_Model import DecisionModel
from PIL import Image
import wandb
from argparse import Namespace


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
# np.random.seed(1)
# random.seed(1)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def read_yaml(fileName):
    """ A function to read YAML file"""
    print('reading file name: ' + fileName)
    with open(fileName) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config

def write_yaml(data,fileName):
    """ A function to write YAML file"""
    with open(fileName+'.yml', 'w') as f:
        yaml.dump(data, f)

def DataLoader_batch_sampler(dataloader_train):
        # random pick should be implemented
        for batch_idx, batch_data in enumerate(dataloader_train):
            batch_image = batch_data[0].to(device)
            batch_target = batch_data[1].to(device)
            break
        return batch_image, batch_target

def main(config):
    
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    write_yaml(config, config.config_dir+'config')
    pprint.pprint(config)
    

    # For fast training.
    cudnn.benchmark = True


    # Data loader.
    celeba_loader = None
    rafd_loader = None
    bdd_loader = None
    bdd100k_loader = None
    MNIST_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.image_size, config.celeba_crop_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers, None, None)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.image_size, config.rafd_crop_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers, None, None)
    if config.dataset in ['BDD']:
        if config.mode=='train':
            bdd_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                    image_size=config.bdd_load_size, crop_size=config.bdd_load_size, batch_size=config.batch_size,
                                    dataset='BDD', mode=config.mode, num_workers=config.num_workers,
                                    image_root=config.bdd_image_root, gt_root_train=config.bdd_gt_root_train)
        else:
            bdd_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                    image_size=config.bdd_load_size, crop_size=config.bdd_load_size, batch_size=config.batch_size,
                                    dataset='BDD', mode=config.mode, num_workers=config.num_workers,
                                    image_root=config.bdd_image_root, gt_root_train=config.bdd_gt_root_val)

    if config.dataset in ['BDD100k']:
        if config.mode=='train':
            bdd100k_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                    image_size=config.bdd_load_size, crop_size=config.bdd_load_size, batch_size=config.batch_size,
                                    dataset='BDD100k', mode=config.mode, num_workers=config.num_workers,
                                    image_root=config.bdd100k_image_root_train, gt_root_train=config.bdd100k_gt_root_train)
        else:
            bdd100k_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                    image_size=config.bdd_load_size, crop_size=config.bdd_load_size, batch_size=config.batch_size,
                                    dataset='BDD100k', mode=config.mode, num_workers=config.num_workers,
                                    image_root=config.bdd100k_image_root_val, gt_root_train=config.bdd100k_gt_root_val)

    if config.dataset in ['MNIST']:
        MNIST_loader = get_loader(image_dir=config.MNIST_image_dir, attr_path=None, selected_attrs=None,
                                 image_size=config.image_size, crop_size=config.MNIST_crop_size, batch_size=config.batch_size,
                                 dataset='MNIST', mode=config.mode, num_workers=config.num_workers,
                                 image_root=None, gt_root_train=None)
    solver = Solver(celeba_loader, rafd_loader, bdd_loader, bdd100k_loader, MNIST_loader, config, device)

        
##################for metrics only#######################
    if "FID" in config.metrics or "KID" in config.metrics:
        mode = 'test' if config.mode=='train' else 'train'
        if config.dataset in ['CelebA', 'Both']:
            data_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                    config.image_size, config.celeba_crop_size, config.batch_size,
                                    'CelebA', mode, config.num_workers, None, None)
        if config.dataset in ['RaFD', 'Both']:
            data_loader = get_loader(config.rafd_image_dir, None, None,
                                    config.image_size, config.rafd_crop_size, config.batch_size,
                                    'RaFD', mode, config.num_workers, None, None)
        if config.dataset in ['BDD']:
            if mode=='train':
                data_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                        image_size=config.bdd_load_size, crop_size=config.bdd_load_size,  batch_size=config.batch_size,
                                        dataset='BDD', mode=config.mode, num_workers=config.num_workers,
                                        image_root=config.bdd_image_root, gt_root_train=config.bdd_gt_root_train)
            else:
                data_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                        image_size=config.bdd_load_size, crop_size=config.bdd_load_size, batch_size=config.batch_size,
                                        dataset='BDD', mode=config.mode, num_workers=config.num_workers,
                                        image_root=config.bdd_image_root, gt_root_train=config.bdd_gt_root_val)
        if config.dataset in ['BDD100k']:
            if mode=='train':
                data_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                        image_size=config.bdd_load_size, crop_size=config.bdd_load_size, batch_size=config.batch_size,
                                        dataset='BDD100k', mode=config.mode, num_workers=config.num_workers,
                                        image_root=config.bdd100k_image_root_train, gt_root_train=config.bdd100k_gt_root_train)
            else:
                data_loader = get_loader(image_dir=None, attr_path=None, selected_attrs=None,
                                        image_size=config.bdd_load_size, crop_size=config.bdd_load_size, batch_size=config.batch_size,
                                        dataset='BDD100k', mode=config.mode, num_workers=config.num_workers,
                                        image_root=config.bdd100k_image_root_val, gt_root_train=config.bdd100k_gt_root_val)
        if config.dataset in ['MNIST']:
            data_loader = get_loader(image_dir=config.MNIST_image_dir, attr_path=None, selected_attrs=None,
                                    image_size=config.image_size, crop_size=config.MNIST_crop_size, batch_size=config.batch_size,
                                    dataset='MNIST', mode=mode, num_workers=config.num_workers,
                                    image_root=None, gt_root_train=None)

        for i, batch_data in enumerate(data_loader):
            x_real,_ = batch_data
            x_real = x_real.to(device)

            x_real_normal = ((x_real-x_real.min()) / (x_real.max()-x_real.min()))
            if "FID" in config.metrics:
                solver.fid.update(x_real, real=True)
            if "KID" in config.metrics:
                solver.kid.update(x_real, real=True)
            if (i == 300):
                break
        del data_loader
####################################################################
    # Solver for training and testing StarGAN.
    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD', 'BDD', 'BDD100k', 'MNIST']: 
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD', 'BDD', 'BDD100k', 'MNIST']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ymlName', type=str, default='CF_starGAN-BDD100k-GradCAM-config.yml', help='yml config name') 
    # CF_attGAN 'sailencyGAN_v3-MNIST-GradCAM-config.yml'
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec_x', type=float, default=10, help='weight for reconstruction loss of image')
    parser.add_argument('--lambda_rec_sal', type=float, default=10, help='weight for reconstruction loss of saliency')
    parser.add_argument('--lambda_sal_fuse', type=float, default=5, help='weight for reconstruction loss of saliency')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--g_loss_cls_of_d', type=bool, default=True, help='generator sees class prediction of D or decision model, true for D')
    parser.add_argument('--g_loss_sal_rec_method', type=str, default='mean', choices=['mean', 'bce'], help='loss between reconstructed saliency and real one')
    parser.add_argument('--saliency_method', type=str, default='GradCAM', help=['AblationCAM', 'GradCAM', 'HiResCAM'])

    # StarGAN and attGAN defaults coef values:  lambda_cls=1, lambda_rec=10, lambda_gp=10,  g_loss_cls_of_d=True,
    #                                           lambda_rec_x=10, lambda_rec_sal=10, g_loss_sal_rec_method=mean, saliency_method=GradCAM

    name = parser.parse_args()
    
    my_config = read_yaml('configs/'+name.ymlName)
    
    my_config.lambda_cls = name.lambda_cls
    my_config.lambda_rec_x = name.lambda_rec_x
    my_config.lambda_rec_sal = name.lambda_rec_sal
    my_config.lambda_sal_fuse = name.lambda_sal_fuse
    my_config.lambda_gp = name.lambda_gp
    my_config.g_loss_cls_of_d = name.g_loss_cls_of_d
    my_config.g_loss_sal_rec_method = name.g_loss_sal_rec_method
    my_config.saliency_method = name.saliency_method


    main(my_config)
    # wandb sweep sweep_grid_bayes.yaml
    # wandb agent amirsamadi/saliency_guided_CF/lvm72b2t


######### To do list: #########
# change the g_loss_rec to BCE
# distinguish g_loss_rec for reconstructed saliency and x and put a coefficent for each
# change the g_loss_cls_of_d and check the resualt
# run the code for MNIST data set so that can have very fast resualts 
# check all saliency generators performance
# check if better target for saliency generation on the decision model's layers can be finded
# check metrics on MNIST to regulate the parameters with using wandb, so the wandb recieves gan performance
#        metrics an regulate the parameters
# ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
# ['move forward', 'stop/slow down', 'turn left', 'turn right']
# analyze the tensorboard results
# decision model needs to use real label, that's why I have to use BDDIO dataset, however for training GAN
#        the label are coming from decision model so for GAN model I can use original BDD dataset which is
#        substantially larger 
# saving decison model's label and using them istead of fetching each time could boost GAN learning
#        processing time 
# 
# 
# 
# 
# 
# 
# 

import pandas as pd 
import wandb

def export_wandb_runs():
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs("warwick/saliency GAN")

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })

    runs_df.to_csv("project.csv")