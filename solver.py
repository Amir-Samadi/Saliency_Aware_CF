from model import Generator
import gc
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import pickle
import csv
import datetime
from Decision_Model import DecisionModel
from pytorch_gan_metrics import get_inception_score, get_fid 
from torchmetrics.image.kid import KernelInceptionDistance
from pytorch_gan_metrics.calc_fid_stats import calc_and_save_stats
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore

import pynvml

def get_memory_free_MiB():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(0))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2



class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, bdd_loader, bdd100k_loader, MNIST_loader, config, device):
        """Initialize configurations."""

        # wandb
        self.wandb = config.wandb
        if (self.wandb):
            wandb.init(config=config, entity='amirsamadi', project='saliency_guided_CF')
            # config = wandb.config
            print("lambda_cls:", config.lambda_cls)
            print("lambda_gp:", config.lambda_gp)
            print("lambda_rec_x:", config.lambda_rec_x)
            print("lambda_rec_sal:", config.lambda_rec_sal)
            print("g_loss_cls_of_d:", config.g_loss_cls_of_d)
            print("g_loss_sal_rec_method:", config.g_loss_sal_rec_method)
            print("saliency_method:", config.saliency_method)



        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        self.bdd_loader = bdd_loader
        self.bdd100k_loader = bdd100k_loader
        self.MNIST_loader = MNIST_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_sal_fuse = config.lambda_sal_fuse
        self.lambda_rec_x = config.lambda_rec_x
        self.lambda_rec_sal = config.lambda_rec_sal
        self.lambda_gp = config.lambda_gp
        self.g_loss_cls_of_d = config.g_loss_cls_of_d
        self.g_loss_sal_rec_method =  config.g_loss_sal_rec_method

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        # self.use_tensorboard = config.use_tensorboard
        self.device = device

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.metrics_dir = config.metrics_dir
        self.config_dir = config.config_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        
        #counterfactual
        self.CF_method = config.CF_method 
        # if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2", "sailencyGAN_v3", "CF_attGAN", "attGAN", "starGAN", "CF_starGAN"]):
        if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2", "sailencyGAN_v3", "CF_attGAN", "CF_starGAN"]):

            if config.dataset in ['CelebA']:
                self.decision_model = DecisionModel(opt=config, data_loader_train=self.celeba_loader, 
                                            data_loader_val=self.celeba_loader, device=self.device)
            elif config.dataset in ['RaFD']:
                self.decision_model = DecisionModel(opt=config, data_loader_train=self.rafd_loader,
                                                data_loader_val=self.rafd_loader, device=self.device)
            elif config.dataset in ['BDD', 'BDD100k']:
                self.decision_model = DecisionModel(opt=config, data_loader_train=self.bdd_loader,
                                                data_loader_val=self.bdd_loader, device=self.device)
            elif config.dataset in ['MNIST']:
                self.decision_model = DecisionModel(opt=config, data_loader_train=self.MNIST_loader,
                                                data_loader_val=self.MNIST_loader, device=self.device)

            self.decision_model_LOG_DIR = config.decision_model_LOG_DIR

            if not os.path.exists(config.decision_model_LOG_DIR):
                os.mkdir(config.decision_model_LOG_DIR)
                self.decision_model.train()
            elif (config.decision_model_train == 'train'):
                self.decision_model.train()
            elif (config.decision_model_train == 'test'):
                self.decision_model.model.load_state_dict(torch.load(os.path.join(config.decision_model_LOG_DIR, config.decision_model_name, 'checkpoint.tar'), map_location=device)['model_state_dict'])
                self.decision_model.test() 
            else:
                self.decision_model.model.load_state_dict(torch.load(os.path.join(config.decision_model_LOG_DIR, config.decision_model_name, 'checkpoint.tar'), map_location=device)['model_state_dict'])


            # self.decision_model.model.eval()


        # Build the model and tensorboard.
        self.build_model()
        # if self.use_tensorboard:
        #     self.build_tensorboard()
        
        # metrics
        self.metric_step = config.metric_step
        self.metrics = config.metrics
        
        if "KID" in self.metrics:
            # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
            self.kid = KernelInceptionDistance(subset_size=self.batch_size , reset_real_features=True, normalize=True).to(self.device)
        if "FID" in self.metrics:
            # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
            self.fid = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=True).to(self.device)
            self.fid.reset()
        if "LPIPS" in self.metrics:
            # If set to True will instead expect input to be in the [0,1] range.
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)
        if "IS" in self.metrics:
            # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
            self.inception = InceptionScore(normalize=True).to(self.device)

        # if not os.path.exists(os.path.join(self.metrics_dir, 'FID/', self.dataset, 'statistics.npz')):
        #     if config.dataset in ['CelebA']:
        #         calc_and_save_stats( data_loader = self.celeba_loader, 
        #                         output_path = os.path.join(self.metrics_dir, 'FID/', self.dataset, 'statistics.npz'),\
        #                         batch_size = self.batch_size, img_size = config.image_size,\
        #                         use_torch = torch.cuda.is_available())
        #     elif config.dataset in ['RaFD']:
        #         calc_and_save_stats( data_loader = self.rafd_loader,   
        #                         output_path = os.path.join(self.metrics_dir, 'FID/', self.dataset, 'statistics.npz'),\
        #                         batch_size = self.batch_size, img_size = config.image_size,\
        #                         use_torch = torch.cuda.is_available())
        #     elif config.dataset in ['BDD']:
        #         calc_and_save_stats(data_loader = self.bdd_loader,
        #                         output_path = os.path.join(self.metrics_dir, 'FID/', self.dataset, 'statistics.npz'),\
        #                         batch_size = self.batch_size, img_size = config.image_size,\
        #                         use_torch = torch.cuda.is_available())
        #     elif config.dataset in ['MNIST']:
        #         calc_and_save_stats(data_loader = self.MNIST_loader,
        #                         output_path = os.path.join(self.metrics_dir, 'FID/', self.dataset, 'statistics.npz'),\
        #                         batch_size = self.batch_size, img_size = config.image_size,\
        #                         use_torch = torch.cuda.is_available())



    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD', 'BDD', 'BDD100k', 'MNIST']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num, self.CF_method)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num, self.CF_method)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    
    def get_validity(self, fake, c_trg):
        with torch.no_grad():
            fake_labels = self.decision_model.decisionModel_label_gen(fake, self.dataset)
            valid_targets = torch.all(torch.eq(fake_labels, c_trg), dim=1) 
            return (valid_targets.sum()/fake_labels.size(0)).tolist()

    def get_sparcity(self, x_real, fake):
        return (fake != x_real).type(torch.float32).mean()
    
    def get_mean_dis(self, x_real, fake):
        return F.l1_loss(fake, x_real).item()

    # def build_tensorboard(self):
    #     """Build a tensorboard logger."""
    #     from logger import Logger
    #     self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim).to(self.device)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            elif dataset in ['BDD', 'BDD100k', 'MNIST']:
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
        elif dataset in ['BDD', 'BDD100k', 'MNIST']:
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'BDD100k':
            data_loader = self.bdd100k_loader
        elif self.dataset == 'BDD':
            data_loader = self.bdd_loader
        elif self.dataset == 'MNIST':
            data_loader = self.MNIST_loader


        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2","sailencyGAN_v3", "CF_attGAN", "CF_starGAN"]):  
            c_org = self.decision_model.decisionModel_label_gen(x_fixed, self.dataset)
            c_org_fixed = c_org.to(self.device)
        c_trg_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)


        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # metrics
        if "KID" in self.metrics:
            KID_list, KID_std_list = [], []
        if "FID" in self.metrics:
            FID_list = []
        if "LPIPS" in self.metrics:
            LPIPS_list = []
        if "IS" in self.metrics:
            IS_list, IS_std_list = [], []
        if "Sparsity" in self.metrics:
            Sparsity_list = []
        if "Validity" in self.metrics and (self.CF_method in ["sailencyGAN", "sailencyGAN_v3", "sailencyGAN_v2"]):
            Validity_list = []

        # Start training.
        print('Start training...')
        start_time = time.time()

        metrics = {}
        for i in range(start_iters, self.num_iters):
            # gc.collect()

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
 
            # Fetch real images and labels.
            if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2", "sailencyGAN_v3", "CF_attGAN", "CF_starGAN"]):
                try:
                    x_real = next(data_iter)[0]
                    x_real = x_real.to(self.device)
                    label_org = self.decision_model.decisionModel_label_gen(x_real, self.dataset)
                except:
                    data_iter = iter(data_loader)
                    x_real = next(data_iter)[0]
                    x_real = x_real.to(self.device)
                    label_org = self.decision_model.decisionModel_label_gen(x_real, self.dataset)
            elif (self.CF_method in ["attGAN", "starGAN"]):
                try:
                    x_real, label_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, label_org = next(data_iter)
            else:
                print("the counterfactual method is not implemented")


            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset in ['CelebA', 'BDD', 'BDD100k', 'MNIST']:
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
            
            if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2", "sailencyGAN_v3"]):
                saliency_real = self.decision_model.saliency_gen(x_real, c_org, c_trg)
                saliency_real = saliency_real.to(self.device)


            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            
            # Compute loss with fake images.
            if (self.CF_method in ["sailencyGAN", "sailencyGAN_v3"]):
                x_fake, attention_mask, content_mask, _ = self.G(x_real, c_trg, saliency_real)
            elif (self.CF_method in ["attGAN", "CF_attGAN"]):
                x_fake, attention_mask, content_mask = self.G(x_real, c_trg, None)
            elif (self.CF_method in ["starGAN", "CF_starGAN"]):    
                x_fake = self.G(x_real, c_trg, None)
            elif (self.CF_method=="sailencyGAN_v2"):    
                x_fake, saliency_fake, content_mask = self.G(x_real, c_trg, saliency_real)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            
            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                if (self.CF_method in ["sailencyGAN", "sailencyGAN_v3"]):
                    x_fake, attention_mask, content_mask, saliency_fake = self.G(x_real, c_trg, saliency_real)
                elif (self.CF_method in ["attGAN", "CF_attGAN"]):
                    x_fake, attention_mask, content_mask = self.G(x_real, c_trg, None)
                elif (self.CF_method in ["starGAN", "CF_starGAN"]):
                    x_fake = self.G(x_real, c_trg, None)
                elif (self.CF_method=="sailencyGAN_v2"):    
                    x_fake, saliency_fake, content_mask = self.G(x_real, c_trg, saliency_real)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                if (self.g_loss_cls_of_d):
                    g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)
                else:
                    decisionModel_Cls = self.decision_model.decisionModel_label_gen(x_fake, self.dataset)
                    g_loss_cls = self.classification_loss(decisionModel_Cls, label_trg, self.dataset)

                # Target-to-original domain.
                if (self.CF_method in ["sailencyGAN", "sailencyGAN_v3"]):
                    x_reconst, _, _, saliency_reconst = self.G(x_fake, c_org, saliency_fake)
                elif (self.CF_method in ["attGAN", "CF_attGAN"]):
                    x_reconst, _, _ = self.G(x_fake, c_org, None)
                elif (self.CF_method in ["starGAN", "CF_starGAN"]):
                    x_reconst = self.G(x_fake, c_org, None)
                elif (self.CF_method=="sailencyGAN_v2"):
                    x_reconst, saliency_reconst, _ = self.G(x_fake, c_org, saliency_fake)
                    
                g_loss_rec_x = F.l1_loss(x_reconst, x_real)
                if(self.CF_method in ["sailencyGAN", "sailencyGAN_v3", "sailencyGAN_v2"]):
                    if (self.g_loss_sal_rec_method == 'mean'):
                        g_loss_rec_sal = F.l1_loss(saliency_reconst, saliency_real)

                        ##loss for fusing saliency
                        loss_sal_fuse = F.l1_loss   (x_fake * (1-saliency_real.mean(dim=1).unsqueeze(1)),\
                                                    (x_real * (1-saliency_real.mean(dim=1).unsqueeze(1)))) 

                    elif (self.g_loss_sal_rec_method == 'bce'):
                        g_loss_rec_sal = F.binary_cross_entropy_with_logits(saliency_reconst,
                                                                             (saliency_real>0.5).float())

                        ##loss for fusing saliency
                        loss_sal_fuse = F.binary_cross_entropy_with_logits  (x_fake * (1-saliency_real.mean(dim=1).unsqueeze(1)),\
                                                                            (x_real * (1-saliency_real.mean(dim=1).unsqueeze(1)))) 


                    g_loss_rec = self.lambda_rec_x * g_loss_rec_x + self.lambda_rec_sal * g_loss_rec_sal
                    g_loss = g_loss_fake + g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_sal_fuse*loss_sal_fuse
                else:
                    g_loss_rec = self.lambda_rec_x * g_loss_rec_x
                    g_loss = g_loss_fake + g_loss_rec + self.lambda_cls * g_loss_cls

                ########

                # loss_sal_fuse = ((x_fake - x_real) * (1-saliency_real)).abs().sum()
                # loss_sal_fuse = (((x_fake - x_real)!=0) * (1-saliency_real)).sum()
                ########
                # Backward and optimize.
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_x_rec'] = g_loss_rec_x.item()
                if(self.CF_method in ["sailencyGAN", "sailencyGAN_v3", "sailencyGAN_v2"]):
                    loss['G/loss_sal_rec'] = g_loss_rec_sal.item()
                    loss['G/loss_sal_fuse'] = loss_sal_fuse.item()
                else:
                    loss['G/loss_sal_rec']  = 0
                    loss['G/loss_sal_fuse'] = 0

                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. metrics                              #
            # =================================================================================== #
            
            if "KID" in self.metrics:
                x_real = ((x_real-x_real.min()) / (x_real.max()-x_real.min()))*2-1 
                x_fake = ((x_fake-x_fake.min()) / (x_fake.max()-x_fake.min()))*2-1 
                self.kid.update(x_fake, real=False)
                loss['metric/KID'], loss['metric/KID_std'] = self.kid.compute()
            if "FID" in self.metrics:
                self.fid.update(x_fake, real=False)
                loss['metric/FID'] = self.fid.compute().item()
            if "LPIPS" in self.metrics:
                loss['metric/lpips'] = self.lpips(x_real, x_fake).item() 
            if "IS" in self.metrics:
                self.inception.update(x_fake)
                inceptionScore = self.inception.compute()
                loss['metric/IS'] = inceptionScore[0].item() 
                loss['metric/IS_std'] = inceptionScore[1].item() 
            if "Sparsity" in self.metrics:
                loss['metric/sparsity'] = self.get_sparcity(x_real, x_fake)               
            if "mean_dis" in self.metrics:
                loss['metric/mean_dis'] = self.get_mean_dis(x_real, x_fake)               
            if "Validity" in self.metrics and (self.CF_method in ["sailencyGAN", "sailencyGAN_v3", "sailencyGAN_v2"]):
                loss['metric/validity'] = self.get_validity(x_fake, c_trg)
            
            # =================================================================================== #
            #                                 5. Miscellaneous                                    #
            # =================================================================================== #


            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                # if self.use_tensorboard:
                #     for tag, value in loss.items():
                #         self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                # GPU_memory = get_memory_free_MiB()
                
                x_fake_list = [x_fixed]
                x_attention_list = [x_fixed]
                x_content_list = [x_fixed]
                if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2", "sailencyGAN_v3"]): 
                    x_saliency_list = [x_fixed]
                for indx, c_fixed in enumerate (c_trg_fixed_list):
                    # x_fake_list.append(self.G(x_fixed, c_fixed))
                    if (self.CF_method=="sailencyGAN"):
                        saliency_fixed = self.decision_model.saliency_gen(x_fixed, c_org_fixed[indx], c_fixed)
                        saliency_fixed = saliency_fixed.to(self.device)
                        with torch.no_grad():
                            fake, attention, content, saliency = self.G(x_fixed, c_fixed, saliency_fixed)
                        attention = (attention - 0.5) / 0.5
                        x_fake_list.append(fake)
                        x_attention_list.append(attention)
                        x_content_list.append(content)
                        x_saliency_list.extend([saliency, saliency_fixed])
                    elif (self.CF_method in ["attGAN", "CF_attGAN"]):
                        with torch.no_grad():
                            fake, attention, content = self.G(x_fixed, c_fixed, None)
                        attention = (attention - 0.5) / 0.5
                        x_fake_list.append(fake)
                        x_attention_list.append(attention)
                        x_content_list.append(content)
                    elif (self.CF_method in ["starGAN", "CF_starGAN"]):
                        fake = self.G(x_fixed, c_fixed, None)
                        x_fake_list.append(fake)
                    elif (self.CF_method=="sailencyGAN_v2"):
                        saliency_fixed = self.decision_model.saliency_gen(x_fixed, c_org_fixed[indx], c_fixed)
                        saliency_fixed = saliency_fixed.to(self.device)
                        with torch.no_grad():
                            fake,  saliency, content = self.G(x_fixed, c_fixed, saliency_fixed)
                        x_fake_list.append(fake)
                        x_content_list.append(content)
                        x_saliency_list.extend([saliency, saliency_fixed])
                    elif (self.CF_method=="sailencyGAN_v3"):
                        saliency_fixed = self.decision_model.saliency_gen(x_fixed, c_org_fixed, c_fixed)
                        saliency_fixed = saliency_fixed.to(self.device)
                        with torch.no_grad():
                            fake, attention, content, saliency = self.G(x_fixed, c_fixed, saliency_fixed)
                        # attention = (attention - 0.5) / 0.5
                        x_fake_list.append(fake)
                        x_attention_list.append(attention)
                        x_content_list.append(content)
                        x_saliency_list.append(saliency.mean(dim=1).unsqueeze(1).repeat(1, 3, 1, 1))
                        x_saliency_list.append(saliency_fixed.mean(dim=1).unsqueeze(1).repeat(1, 3, 1, 1))
                        # saliency_fixed_u = [i.unsqueeze(dim=1) for i in saliency_fixed.unbind(dim=1)][0]
                        # saliency_u = [i.unsqueeze(dim=1) for i in saliency.unbind(dim=1)][0]
                        # x_saliency_list.extend([saliency_u, saliency_fixed_u])
                            
                    
                    if (self.CF_method=="sailencyGAN"):
                        x_concat = torch.cat(x_fake_list, dim=3)
                        attention_concat = torch.cat(x_attention_list, dim=3)
                        content_concat = torch.cat(x_content_list, dim=3)
                        saliency_concat = torch.cat(x_saliency_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                        attention_path = os.path.join(self.sample_dir, '{}-attention.jpg'.format(i+1))
                        content_path = os.path.join(self.sample_dir, '{}-content.jpg'.format(i+1))
                        saliency_path = os.path.join(self.sample_dir, '{}-saliency.jpg'.format(i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        save_image(self.denorm(attention_concat.data.cpu()), attention_path, nrow=1, padding=0)
                        save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
                        save_image(self.denorm(saliency_concat.data.cpu()), saliency_path, nrow=1, padding=0)
                    elif (self.CF_method in ["attGAN", "CF_attGAN"]):
                        x_concat = torch.cat(x_fake_list, dim=3)
                        attention_concat = torch.cat(x_attention_list, dim=3)
                        content_concat = torch.cat(x_content_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                        attention_path = os.path.join(self.sample_dir, '{}-attention.jpg'.format(i+1))
                        content_path = os.path.join(self.sample_dir, '{}-content.jpg'.format(i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        save_image(self.denorm(attention_concat.data.cpu()), attention_path, nrow=1, padding=0)
                        save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
                    elif (self.CF_method in ["starGAN", "CF_starGAN"]):
                        x_concat = torch.cat(x_fake_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    elif (self.CF_method=="sailencyGAN_v2"):
                        x_concat = torch.cat(x_fake_list, dim=3)
                        content_concat = torch.cat(x_content_list, dim=3)
                        saliency_concat = torch.cat(x_saliency_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                        content_path = os.path.join(self.sample_dir, '{}-content.jpg'.format(i+1))
                        saliency_path = os.path.join(self.sample_dir, '{}-saliency.jpg'.format(i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
                        save_image(self.denorm(saliency_concat.data.cpu()), saliency_path, nrow=1, padding=0)
                    elif (self.CF_method=="sailencyGAN_v3"):
                        x_concat = torch.cat(x_fake_list, dim=3)
                        attention_concat = torch.cat(x_attention_list, dim=3)
                        content_concat = torch.cat(x_content_list, dim=3)
                        saliency_concat = torch.cat(x_saliency_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                        attention_path = os.path.join(self.sample_dir, '{}-attention.jpg'.format(i+1))
                        content_path = os.path.join(self.sample_dir, '{}-content.jpg'.format(i+1))
                        saliency_path = os.path.join(self.sample_dir, '{}-saliency.jpg'.format(i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        save_image(self.denorm(attention_concat.data.cpu()), attention_path, nrow=1, padding=0)
                        save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
                        save_image(self.denorm(saliency_concat.data.cpu()), saliency_path, nrow=1, padding=0)

                    print('Saved real and fake images into {}...'.format(sample_path))
                            # save metrics
                # if (i+1) % self.metric_step == 0:
                #     fake = torch.cat(x_fake_list[1:], dim=0)
                #     x_real = x_fixed.repeat(len(c_trg_fixed_list),1,1,1)
                #     c_trg = torch.cat(c_trg_fixed_list, dim=0)
                #     if 'IS' in self.metrics:
                #         IS_metric, IS_std_metric = get_inception_score(fake) # Inception Score
                #         IS_list.append(IS_metric)
                #         IS_std_list.append(IS_std_metric)
                #         loss['metric/IS'] = IS_list[-1]
                #         loss['metric/IS_std_list'] = IS_std_list[-1]
                #         # loss['metric/IS'] = Average(IS_list)
                #         # loss['metric/IS_std_list'] = Average(IS_std_list)
                #         # print("Inception Score", Average(IS_list))
                #         # print("Inception Score_SDT", Average(IS_std_list))
                #     if 'FID' in self.metrics:
                #         self.fid.compute()
                #         FID_list.append(get_fid(fake, os.path.join(self.metrics_dir,
                #                 'FID/', self.dataset, 'statistics.npz'))) # Frechet Inception Distance
                #         loss['metric/FID'] = FID_list[-1]
                #         # loss['metric/FID'] = Average(FID_list)
                #         # print("Frechet Inception Distance", Average(FID_list))
                #     if 'sparcity' in self.metrics:
                #         saprcity_list.append(self.get_sparcity(x_real, fake))
                #         loss['metric/saprcity'] = saprcity_list[-1]
                #         # loss['metric/saprcity'] = Average(saprcity_list)
                #         # print("saprcity", Average(saprcity_list))
                #     if 'validity' in self.metrics:
                #         validity_list.append(self.get_validity(fake, c_trg)) #for counterfactual only
                #         loss['metric/validity'] = validity_list[-1]
                #         # loss['metric/validity'] = Average(validity_list)
                #         # print("validity", Average(validity_list))
                #     # wandb
                # print("GPU memory used by sample process: ", GPU_memory - get_memory_free_MiB())
                if(self.wandb):
                    metrics = {
                        'G/loss_fake': loss['G/loss_fake'],
                        'G/loss_rec':loss['G/loss_rec'],
                        'G/loss_x_rec': loss['G/loss_x_rec'],
                        'G/loss_cls': loss['G/loss_cls'],
                        'G/loss_sal_fuse': loss['G/loss_sal_fuse'],
                        'G/loss_sal_rec': loss['G/loss_sal_rec'], 
                        'D/loss_real': loss['D/loss_real'], 
                        'D/loss_fake': loss['D/loss_fake'], 
                        'D/loss_cls': loss['D/loss_cls'], 
                        'D/loss_gp': loss['D/loss_gp'] 
                            }
                #     # GPU_memory = get_memory_free_MiB()
                    if "KID" in self.metrics:
                        metrics['metric/KID'] = loss['metric/KID']
                        metrics['metric/KID_std'] = loss['metric/KID_std']
                    if "FID" in self.metrics:
                        metrics['metric/FID'] = loss['metric/FID']
                    if "LPIPS" in self.metrics:
                        metrics['metric/lpips'] = loss['metric/lpips'] 
                    if "IS" in self.metrics:
                        metrics['metric/IS'] = loss['metric/IS']
                        metrics['metric/IS_std'] = loss['metric/IS_std']
                    if "Sparsity" in self.metrics:
                        metrics['metric/sparsity'] = loss['metric/sparsity']                                           
                    if "mean_dis" in self.metrics:
                        metrics['metric/mean_dis'] = loss['metric/mean_dis']                                           
                    if "Validity" in self.metrics and (self.CF_method in ["sailencyGAN", "sailencyGAN_v3", "sailencyGAN_v2"]):
                        metrics['metric/validity'] = loss['metric/validity']
                        # totall_metric = -loss['metric/IS'] + loss['metric/FID'] - 10*loss['metric/validity'] + 2*loss['metric/sparsity']                            
                    wandb.log(metrics)
                    # print("GPU memory used by WandB log: ", GPU_memory - get_memory_free_MiB())

                    # save_metrics_dir = os.path.join(self.config_dir, 'metrics')
                    # print(save_metrics_dir)
                    # if not os.path.exists(save_metrics_dir):
                    #     os.mkdir(save_metrics_dir)        
                    # save_metrics_dir = os.path.join(self.config_dir, 'train')
                    # print(save_metrics_dir)
                    # if not os.path.exists(save_metrics_dir):
                    #     os.mkdir(save_metrics_dir)        
                    # save_metrics_dir = os.path.join(save_metrics_dir, 'metrics_{}.csv'.format(i+1))
                    # print(save_metrics_dir)
                    # if self.metrics:
                    #     metrics_value = {}
                    #     if 'IS' in self.metrics:
                    #         metrics_value['IS'] = IS_list[-1]
                    #         metrics_value['IS_std'] = IS_std_list[-1]
                    #         metrics_value['IS_list'] = IS_list
                    #         metrics_value['IS_list'] = IS_std_list
                    #     if 'FID' in self.metrics:
                    #         metrics_value['FID'] = FID_list[-1]
                    #         metrics_value['FID_list'] = FID_list
                    #     if 'sparcity' in self.metrics:
                    #         metrics_value['saprcity'] = saprcity_list[-1]
                    #         metrics_value['saprcity_list'] = saprcity_list
                    #     if 'validity' in self.metrics:
                    #         metrics_value['validity'] = validity_list[-1]
                    #         metrics_value['validity_list'] = validity_list
                    #     with open(save_metrics_dir, 'w') as f:
                    #         w = csv.DictWriter(f, metrics_value.keys())
                    #         w.writeheader()
                    #         w.writerow(metrics_value)


            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                # GPU_memory = get_memory_free_MiB()
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
                # print("GPU memory used by WandB log: ", GPU_memory - get_memory_free_MiB())

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            


    def test(self):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'BDD':
            data_loader = self.bdd_loader
        elif self.dataset == 'BDD100k':
            data_loader = self.bdd100k_loader
        elif self.dataset == 'MNIST':
            data_loader = self.MNIST_loader
            
        metrics_dict = {}
        wandb_metrics_dict = {}
        if "KID" in self.metrics:
            metrics_dict['metric/KID'] = []
            metrics_dict['metric/KID_std'] = []
        if "FID" in self.metrics:
            metrics_dict['metric/FID'] = []
        if "LPIPS" in self.metrics:
            metrics_dict['metric/LPIPS'] = []
        if "IS" in self.metrics:
            metrics_dict['metric/IS'] = []
            metrics_dict['metric/IS_std'] = []
        if "sparsity" in self.metrics:
            metrics_dict['metric/sparsity'] = []
        if "mean_dis" in self.metrics:
            metrics_dict['metric/mean_dis'] = []
        if "validity" in self.metrics:
            metrics_dict['metric/validity'] = []

        print("there are {} batched data in the data_loader".format(len(data_loader)))
        for i, batch_data in enumerate(data_loader):
            # (x_real, c_org) = list(map(batch_data.get, ['image', 'target']))
            x_real, c_org = batch_data

            # Prepare input images and target domain labels.
            x_real = x_real.to(self.device)
            c_org = c_org.to(self.device)
            if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2", "sailencyGAN_v3", "CF_attGAN"]):
                c_org = self.decision_model.decisionModel_label_gen(x_real, self.dataset) 
            
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
            
            # Translate images.
            x_fake_list = [x_real]
            x_attention_list = [x_real]
            x_content_list = [x_real]
            if (self.CF_method in ["sailencyGAN", "sailencyGAN_v2", "sailencyGAN_v3"]):
                x_saliency_list = [x_real]
            for c_trg in c_trg_list:
                if (self.CF_method=="sailencyGAN"):
                    saliency_real = self.decision_model.saliency_gen(x_real, c_org, c_trg)
                    saliency_real = saliency_real.to(self.device)
                    with torch.no_grad(): 
                        fake, attention, content, saliency = self.G(x_real, c_trg, saliency_real)
                    attention = (attention - attention.min()) / (attention.max()-attention.min())
                    x_fake_list.append(fake)
                    x_attention_list.append(attention)
                    x_content_list.append(content)
                    x_saliency_list.extend([saliency, saliency_real])
                elif (self.CF_method in ["attGAN", "CF_attGAN"]):
                    with torch.no_grad():
                        fake, attention, content = self.G(x_real, c_trg, None)
                    attention = (attention - attention.min()) / (attention.max()-attention.min())
                    x_fake_list.append(fake)
                    x_attention_list.append(attention)
                    x_content_list.append(content)
                elif (self.CF_method in ["starGAN", "CF_starGAN"]):
                    with torch.no_grad():
                        fake = self.G(x_real, c_trg, None)
                    x_fake_list.append(fake)
                elif (self.CF_method=="sailencyGAN_v2"):
                    saliency_real = self.decision_model.saliency_gen(x_real, c_org, c_trg)
                    saliency_real = saliency_real.to(self.device)
                    with torch.no_grad():
                        fake, saliency, content  = self.G(x_real, c_trg, saliency_real)
                    x_fake_list.append(fake)
                    x_content_list.append(content)
                    x_saliency_list.extend([saliency, saliency_real])
                elif (self.CF_method=="sailencyGAN_v3"):
                    saliency_real = self.decision_model.saliency_gen(x_real, c_org, c_trg)
                    saliency_real = saliency_real.to(self.device)
                    with torch.no_grad():
                        fake, attention, content, saliency = self.G(x_real, c_trg, saliency_real)
                    attention = (attention - attention.min()) / (attention.max()-attention.min())
                    x_fake_list.append(fake)
                    x_attention_list.append(attention)
                    x_content_list.append(content)
                    x_saliency_list.append(saliency_real.mean(dim=1).unsqueeze(1).repeat(1, 3, 1, 1))
                    x_saliency_list.append(saliency.mean(dim=1).unsqueeze(1).repeat(1, 3, 1, 1))
                    # saliency_fixed_u = [i.unsqueeze(dim=1) for i in saliency_real.unbind(dim=1)][0]
                    # saliency_u = [i.unsqueeze(dim=1) for i in saliency.unbind(dim=1)][0]
                    # x_saliency_list.extend([saliency_u, saliency_fixed_u])                  
################ =================================================================================== ################
################                                   metrics_value                                     ################
################ =================================================================================== ################
                if "KID" in self.metrics or "FID" in self.metrics or "LPIPS" in self.metrics or "IS" in self.metrics:
                    x_real_normal = ((x_real-x_real.min()) / (x_real.max()-x_real.min()))
                    x_fake_normal = ((fake-fake.min()) / (fake.max()-fake.min()))             
                
                if "KID" in self.metrics:
                    self.kid.update(x_fake_normal, real=False)
                    kidValue = self.kid.compute()
                    metrics_dict['metric/KID'].append(kidValue[0].item())
                    metrics_dict['metric/KID_std'].append(kidValue[1].item())
                    wandb_metrics_dict['metric/KID'] = metrics_dict['metric/KID'][-1]
                    wandb_metrics_dict['metric/KID_std'] = metrics_dict['metric/KID_std'][-1]
               
                if "FID" in self.metrics:
                    self.fid.update(x_fake_normal, real=False)
                    metrics_dict['metric/FID'].append(self.fid.compute().item())
                    wandb_metrics_dict['metric/FID'] = metrics_dict['metric/FID'][-1]
                
                if "LPIPS" in self.metrics:
                    metrics_dict['metric/LPIPS'].append(self.lpips(x_real_normal, x_fake_normal).item()) 
                    wandb_metrics_dict['metric/LPIPS'] = metrics_dict['metric/LPIPS'][-1]
                
                if "IS" in self.metrics:
                    self.inception.update(x_fake_normal)
                    inceptionScore = self.inception.compute()
                    metrics_dict['metric/IS'].append(inceptionScore[0].item()) 
                    metrics_dict['metric/IS_std'].append(inceptionScore[1].item()) 
                    wandb_metrics_dict['metric/IS'] = metrics_dict['metric/IS'][-1] 
                    wandb_metrics_dict['metric/IS_std'] = metrics_dict['metric/IS_std'][-1] 
                
                if "sparsity" in self.metrics:
                    #sparsity should see the actual generated image not normalized one
                    metrics_dict['metric/sparsity'].append(self.get_sparcity(x_real, fake))               
                    # wandb_metrics_dict['metric/sparsity'] = metrics_dict['metric/sparsity'][-1]             
                    wandb_metrics_dict['metric/sparsity'+str(c_trg[0].nonzero().item())] = metrics_dict['metric/sparsity'][-1]             
                
                if "mean_dis" in self.metrics:
                    #sparsity should see the actual generated image not normalized one
                    metrics_dict['metric/mean_dis'].append(self.get_mean_dis(x_real, fake))               
                    # wandb_metrics_dict['metric/mean_dis'] = metrics_dict['metric/mean_dis'][-1]  
                    wandb_metrics_dict['metric/mean_dis'+str(c_trg[0].nonzero().item())] = metrics_dict['metric/mean_dis'][-1]  

                if "validity" in self.metrics and\
                         (self.CF_method in ["sailencyGAN", "sailencyGAN_v3", "sailencyGAN_v2", "CF_attGAN", "attGAN", "CF_starGAN", "starGAN"]):
                    #decision_model should see the actual generated image not normalized one
                    metrics_dict['metric/validity'].append(self.get_validity(fake, c_trg))
                    # wandb_metrics_dict['metric/validity'] = metrics_dict['metric/validity'][-1]
                    wandb_metrics_dict['metric/validity'+str(c_trg[0].nonzero().item())] = metrics_dict['metric/validity'][-1]
                    
                if(self.wandb):
                    wandb.log(wandb_metrics_dict)
################ =================================================================================== ################
################                                 save the results                                    ################
################ =================================================================================== ################
            if (self.CF_method=="sailencyGAN"):
                x_concat = torch.cat(x_fake_list, dim=3)
                attention_concat = torch.cat(x_attention_list, dim=3)
                content_concat = torch.cat(x_content_list, dim=3)
                saliency_concat = torch.cat(x_saliency_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                attention_path = os.path.join(self.result_dir, '{}-attention.jpg'.format(i + 1))
                save_image(self.denorm(attention_concat.data.cpu()), attention_path, nrow=1, padding=0)
                content_path = os.path.join(self.result_dir, '{}-content.jpg'.format(i + 1))
                save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
                saliency_path = os.path.join(self.result_dir, '{}-saliency.jpg'.format(i + 1))
                save_image(self.denorm(saliency_concat.data.cpu()), saliency_path, nrow=1, padding=0)
            elif (self.CF_method in ["attGAN", "CF_attGAN"]):
                x_concat = torch.cat(x_fake_list, dim=3)
                attention_concat = torch.cat(x_attention_list, dim=3)
                content_concat = torch.cat(x_content_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                attention_path = os.path.join(self.result_dir, '{}-attention.jpg'.format(i + 1))
                save_image(self.denorm(attention_concat.data.cpu()), attention_path, nrow=1, padding=0)
                content_path = os.path.join(self.result_dir, '{}-content.jpg'.format(i + 1))
                save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
            elif (self.CF_method in ["starGAN", "CF_starGAN"]):
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            elif (self.CF_method=="sailencyGAN_v2"):
                x_concat = torch.cat(x_fake_list, dim=3)
                content_concat = torch.cat(x_content_list, dim=3)
                saliency_concat = torch.cat(x_saliency_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                content_path = os.path.join(self.result_dir, '{}-content.jpg'.format(i + 1))
                save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
                saliency_path = os.path.join(self.result_dir, '{}-saliency.jpg'.format(i + 1))
                save_image(self.denorm(saliency_concat.data.cpu()), saliency_path, nrow=1, padding=0)
            elif (self.CF_method=="sailencyGAN"):
                x_concat = torch.cat(x_fake_list, dim=3)
                attention_concat = torch.cat(x_attention_list, dim=3)
                content_concat = torch.cat(x_content_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                attention_path = os.path.join(self.result_dir, '{}-attention.jpg'.format(i + 1))
                save_image(self.denorm(attention_concat.data.cpu()), attention_path, nrow=1, padding=0)
                content_path = os.path.join(self.result_dir, '{}-content.jpg'.format(i + 1))
                save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
            elif (self.CF_method=="sailencyGAN_v3"):
                x_concat = torch.cat(x_fake_list, dim=3)
                attention_concat = torch.cat(x_attention_list, dim=3)
                content_concat = torch.cat(x_content_list, dim=3)
                saliency_concat = torch.cat(x_saliency_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                attention_path = os.path.join(self.result_dir, '{}-attention.jpg'.format(i+1))
                save_image(self.denorm(attention_concat.data.cpu()), attention_path, nrow=1, padding=0)
                content_path = os.path.join(self.result_dir, '{}-content.jpg'.format(i+1))
                save_image(self.denorm(content_concat.data.cpu()), content_path, nrow=1, padding=0)
                saliency_path = os.path.join(self.result_dir, '{}-saliency.jpg'.format(i+1))
                save_image(self.denorm(saliency_concat.data.cpu()), saliency_path, nrow=1, padding=0)

            print('Saved real and fake images into {}...'.format(result_path))

        print("average value of metric/FID: {}".format(sum(metrics_dict['metric/FID'])/len(metrics_dict['metric/FID']))) 
        print("average value of metric/KID: {}".format(sum(metrics_dict['metric/KID'])/len(metrics_dict['metric/KID']))) 
        print("average value of metric/KID_std: {}".format(sum(metrics_dict['metric/KID_std'])/len(metrics_dict['metric/KID_std']))) 
        print("average value of metric/LPIPS: {}".format(sum(metrics_dict['metric/LPIPS'])/len(metrics_dict['metric/LPIPS']))) 
        print("average value of metric/IS: {}".format(sum(metrics_dict['metric/IS'])/len(metrics_dict['metric/IS']))) 
        print("average value of metric/IS_std: {}".format(sum(metrics_dict['metric/IS_std'])/len(metrics_dict['metric/IS_std']))) 
        print("average value of metric/sparsity: {}".format(sum(metrics_dict['metric/sparsity'])/len(metrics_dict['metric/sparsity']))) 
        print("average value of metric/mean_dis: {}".format(sum(metrics_dict['metric/mean_dis'])/len(metrics_dict['metric/mean_dis']))) 
        print("average value of metric/validity: {}".format(sum(metrics_dict['metric/validity'])/len(metrics_dict['metric/validity']))) 
        
            # metrics
        #     fake = torch.cat(x_fake_list[1:], dim=0)
        #     x_real = x_real.repeat(10,1,1,1)
        #     c_trg = torch.cat(c_trg_list, dim=0)
        #     if 'IS' in self.metrics:
        #         start.record()
        #         IS_metric, IS_std_metric = get_inception_score(fake) # Inception Score
        #         IS_list.append(IS_metric)
        #         IS_std_list.append(IS_std_metric) 
        #         print("Inception Score", Average(IS_list))
        #         print("Inception Score_SDT", Average(IS_std_list))
        #         end.record()
        #         torch.cuda.synchronize()
        #         print(start.elapsed_time(end))    
        #     if 'FID' in self.metrics:
        #         start.record()
        #         FID_list.append(get_fid(fake, os.path.join(self.metrics_dir,
        #                 'FID/', self.dataset, 'statistics.npz'))) # Frechet Inception Distance
        #         print("Frechet Inception Distance", Average(FID_list))
        #         end.record()
        #         torch.cuda.synchronize()
        #         print(start.elapsed_time(end))    
        #     if 'sparcity' in self.metrics:
        #         start.record()
        #         saprcity_list.append(self.get_sparcity(x_real, fake))
        #         # print("saprcity", Average(saprcity_list))
        #         end.record()
        #         torch.cuda.synchronize()
        #         print(start.elapsed_time(end))    
        #     if 'validity' in self.metrics:
        #         start.record()
        #         validity_list.append(self.get_validity(fake, c_trg)) #for counterfactual only
        #         # print("validity", Average(validity_list))
        #         end.record()
        #         torch.cuda.synchronize()
        #         print(start.elapsed_time(end)) 

        #     save_metrics_dir = os.path.join(self.config_dir, 'metrics/test')
        #     if not os.path.exists(save_metrics_dir):
        #         os.mkdir(save_metrics_dir)        
        #     save_metrics_dir = os.path.join(self.config_dir, 'metrics/test/metrics.csv')
        # if self.metrics:
        #     metrics_value = {}
        #     if 'IS' in self.metrics:
        #         metrics_value['IS_Average'] = Average(IS_list)
        #         metrics_value['IS_std_Average'] = Average(IS_std_list)
        #         metrics_value['IS_list'] = IS_list
        #         metrics_value['IS_list'] = IS_std_list
        #     if 'FID' in self.metrics:
        #         metrics_value['FID_Average'] = Average(FID_list)
        #         metrics_value['FID_list'] = FID_list
        #     if 'sparcity' in self.metrics:
        #         metrics_value['saprcity_Average'] = Average(saprcity_list)
        #         metrics_value['saprcity_list'] = saprcity_list
        #     if 'validity' in self.metrics:
        #         metrics_value['validity_Average'] = Average(validity_list)
        #         metrics_value['validity_list'] = validity_list
        #     with open(save_metrics_dir, 'w') as f:
        #         w = csv.DictWriter(f, metrics_value.keys())
        #         w.writeheader()
        #         w.writerow(metrics_value)

def Average(lst):
    return sum(lst) / len(lst)