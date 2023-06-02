import torch
import torchvision.models as models
from models.DecisionDensenetModel import DecisionDensenetModel
import torch.nn as nn
import os 
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SoftmaxOutputTarget,FasterRCNNBoxScoreTarget,ClassifierOutputSoftmaxTarget,RawScoresOutputTarget,BinaryClassifierOutputTarget\
    ,SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import topk
from torch.nn import functional as F
import cv2
from PIL import Image
from copy import copy
import matplotlib.pyplot as plt
import matplotlib as mpl

class DecisionModel(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, opt, data_loader_train, data_loader_val, device):
        """Initialize configurations."""        
        self.opt = opt
        self.CF_method = opt.CF_method
        self.decision_model_name = opt.decision_model_name 
        self.dataset = opt.dataset
        self.c_dim = opt.c_dim
        self.saliency_method = opt.saliency_method
        self.batch_size = opt.batch_size
        self.device = device
        self.model = DecisionDensenetModel(device=self.device,
            num_classes=len(opt.decision_model_attributes_idx), pretrained=True)
        
        if opt.decision_model_train in ['train', 'test']: 
            self.dataloader_train, self.dataloader_val = data_loader_train, data_loader_val 

        
        if self.opt.decision_model_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=opt.decision_model_lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),lr=opt.decision_model_lr)

        self.criterion = nn.BCELoss(reduction='mean')
        self.model.to(self.device)
        
        self.decision_model_LOG_DIR = opt.decision_model_LOG_DIR
        self.LOG_FOUT = open(os.path.join(opt.decision_model_LOG_DIR, 'log_train.txt'), 'a')
        self.LOG_FOUT.write(str(opt)+'\n')


    def compute_accuracy(self, pred, target):
        same_ids = (pred == target).float().cpu()
        return torch.mean(same_ids,axis=0).numpy()

   
    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        print(out_str)


    def train_one_epoch(self):
        print("Number of batches:", len(self.dataloader_train))
        total_loss = 0
        stat_loss = 0

        total_acc = np.zeros(len(self.opt.decision_model_attributes_idx))
        stat_acc = np.zeros(len(self.opt.decision_model_attributes_idx))

        self.model.train()

        # data_iter = iter(self.dataloader_train) 
        # alaki = next(data_iter)

        for batch_idx, batch_data in enumerate(tqdm(self.dataloader_train)):
            batch_image, batch_target = batch_data
            batch_image = batch_image.to(self.device)
            batch_target = batch_target.to(self.device)
            # Forward pass
            self.optimizer.zero_grad()

            pred = self.model(batch_image)
            pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

            real_labels = torch.index_select(batch_target,
                    1, torch.tensor(self.opt.decision_model_attributes_idx).to(self.device))


            # Compute loss and gradients
            loss = self.criterion(pred,real_labels)
            acc = self.compute_accuracy(pred_labels,real_labels)

            stat_loss += loss.item()
            total_loss += loss.item()
            stat_acc += acc
            total_acc += acc


            loss.backward()
            self.optimizer.step()


            batch_interval = 50
            if (batch_idx+1) % batch_interval == 0:
                self.log_string(' ---- batch: %03d ----' % (batch_idx+1))
                self.log_string('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
                self.log_string('mean accuracy on the last 50 batches: '+
                                 str(stat_acc/batch_interval))
                stat_loss = 0
                stat_acc = 0


        total_mean_loss = total_loss/len(self.dataloader_train)
        total_mean_acc = total_acc/len(self.dataloader_train)
        self.log_string('mean loss over training set: %f'%(total_mean_loss))
        self.log_string('mean accuracy over training set: ' + str(total_mean_acc))

        return total_mean_loss



    def evaluate_one_epoch(self):

        self.model.eval()

        total_loss = 0
        stat_loss = 0
        total_acc = np.zeros(len(self.opt.decision_model_attributes_idx))
        stat_acc = np.zeros(len(self.opt.decision_model_attributes_idx))

        print("Number of batches:", len(self.dataloader_val))

        for batch_idx, batch_data in enumerate(tqdm(self.dataloader_val)):
            batch_image, batch_target = batch_data
            batch_image = batch_image.to(self.device)
            batch_target = batch_target.to(self.device)

            # batch_data['image'] = batch_data['image'].to(self.device)
            # batch_data['target'] = batch_data['target'].to(self.device)

            # Forward pass

            # inputs = batch_data['image']
            with torch.no_grad():
                # pred = self.model(inputs)
                pred = self.model(batch_image)
                pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

                real_labels = torch.index_select(
                    # batch_data['target'],1,torch.tensor(
                    batch_target,1,torch.tensor(
                    self.opt.decision_model_attributes_idx).to(self.device))

            # Compute loss and metrics
            loss = self.criterion(pred,real_labels)
            acc = self.compute_accuracy(pred_labels,real_labels)

            stat_loss += loss.item()
            total_loss += loss.item()
            stat_acc += acc
            total_acc += acc


            batch_interval = 50
            if (batch_idx+1) % batch_interval == 0:
                self.log_string(' ---- batch: %03d ----' % (batch_idx+1))
                self.log_string('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
                self.log_string('mean accuracy on the last 50 batches: ' +
                                 str(stat_acc/batch_interval))
                stat_loss = 0
                stat_acc = 0

        total_mean_loss = total_loss/len(self.dataloader_val)
        total_mean_acc = total_acc/len(self.dataloader_val)

        self.log_string('mean loss over validation set: %f'%(total_mean_loss))
        self.log_string('mean accuracy over validation set: '+str(total_mean_acc))

        return total_mean_loss
    

    def train(self):
        lowest_loss = 100000
        self.log_string("Starting training from the beginning.")

        for epoch in range(self.opt.decision_model_num_epochs):

            # # Train one epoch
            self.log_string(' **** EPOCH: %03d ****' % (epoch+1))
            self.train_one_epoch()

            # Evaluate one epoch
            self.log_string(' **** EVALUATION AFTER EPOCH %03d ****' % (epoch+1))
            total_mean_loss = self.evaluate_one_epoch()
            print("validation loss", total_mean_loss)
            if total_mean_loss < lowest_loss:
                lowest_loss = total_mean_loss
                save_dict = {'epoch': epoch+1, 'optimizer_state_dict': self.optimizer.state_dict(),
                              'loss': total_mean_loss, 'model_state_dict': self.model.state_dict()}
                torch.save(save_dict, os.path.join(self.decision_model_LOG_DIR, self.decision_model_name, 'checkpoint.tar'))
                print("saved new checkpoint")

        del self.dataloader_train
        del self.dataloader_val
    def test(self):
        # Evaluate one epoch
        print("validation loss: ", self.evaluate_one_epoch())

        del self.dataloader_train
        del self.dataloader_val
    
    def decisionModel_label_gen(self, x, datasetName):
        # self.model.eval()

        # decision_model = copy(self.model)
        with torch.no_grad():
            outputs = self.model(x)
        if(datasetName in ['CelebA', 'BDD', 'BDD100k']):        
            class_idx = torch.where(outputs > 0.5 , 1.0, 0.0)
        elif(datasetName in ['RaFD', 'Both', 'MNIST']):
            probs = F.softmax(outputs).data.squeeze()
            # get the class indices of top k probabilities
            class_idx = topk(probs, 1)[1].int().reshape(-1)
            class_idx = F.one_hot(class_idx.type(torch.int64),
                                len(self.opt.decision_model_attributes_idx)).type(torch.float32)
        return class_idx.detach()
    

    def saliency_gen(self, x, org_class, desired_class):
        # self.model.eval()


        # decision_model = copy(self.model.eval())
        # # with torch.no_grad():
        
        # input = torch.zeros_like(x)
        # input.requires_grad=True
        # # x.requires_grad=True
        
        # # with torch.no_grad():

        # preds = decision_model(input)
        # for idx, pred in enumerate(preds):
        #     # loss = self.criterion(pred, desired_class[idx])
        #     loss = self.criterion(pred, org_class[idx])
        #     loss.backward()        
        
        # target_layers = [decision_model.classifier]
        target_layers = [self.model.feat_extract.feat_extract.layer4[-1]]
        # target_layers = [decision_model.feat_extract.feat_extract.features]
        # target_layers = [decision_model.feat_extract.feat_extract.features.denseblock4.denselayer16.conv2]
        if (self.saliency_method == 'GradCAM'):
            cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'HiResCAM'):
            cam = HiResCAM(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'AblationCAM'):
            cam = AblationCAM(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'ScoreCAM'):
            cam = ScoreCAM(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'LayerCAM'):
            cam = LayerCAM(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'FullGrad'):
            cam = FullGrad(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'DeepFeatureFactorization'):
            DeepFeatureFactorization(model=self.model, target_layer=target_layers, computation_on_concepts=decision_model.classifier)
        # cam.batch_size = self.batch_size #??????????????????mishe aya in for loop paien re ba in hal kard?
        
        # fig, ax = plt.subplots(2,3)
        # ax[0,0].imshow(x[0].permute(1,2,0).cpu())
        # ax[0,1].imshow(cam(input_tensor=x, targets=[ClassifierOutputTarget(0)])[0])
        # ax[0,2].imshow(cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0])
        # ax[1,0].imshow(cam(input_tensor=x, targets=[ClassifierOutputTarget(2)])[0])
        # ax[1,1].imshow(cam(input_tensor=x, targets=[ClassifierOutputTarget(3)])[0])
        # ax[1,2].imshow(cam(input_tensor=x, targets=[ClassifierOutputTarget(4)])[0])

        cam.activations_and_grads.model.to(self.device)
        cam.model.to(self.device)
        grayscale_cam = torch.zeros((x.size(0), self.c_dim, x.size(2), x.size(3)))
        toggle_signs = (org_class - desired_class).to(self.device)
        grayscale_cam[torch.where(toggle_signs == -1)] = torch.ones((x.size(2), x.size(3)))
        
        toggle_signs = (toggle_signs > 0).float()


        # if(self.dataset in ['CelebA']):
        #     toggle_signs[:, -2:][torch.where(toggle_signs[:, -2:] == -1)] = 1
        #     toggle_signs = (toggle_signs > 0).float()
        # else:
        #     toggle_signs = (toggle_signs > 0).float()
        
        
        i, j = torch.where(toggle_signs != 0) # i is the images indx and j is the changed feature of the image
        
        if (i.nelement()!=0):
            input_tensor = x[i]
            targets=[ClassifierOutputTarget(category=j[ii]) for ii in range(i.size(0))]
            cams = torch.tensor(cam(input_tensor=input_tensor, targets=targets)).to(self.device)

            
        if (self.CF_method=='sailencyGAN_v3'):
            for image_num in range(self.batch_size):
                update_indicies = torch.where(i==image_num)[0].tolist()
                for update_indice in update_indicies:
                    grayscale_cam[image_num, j[update_indice]] = cams[update_indice] * toggle_signs[image_num, j[update_indice]]

            # fig, ax = plt.subplots(3,6)
            # ax[0,0].imshow(x[0].permute(1,2,0).cpu())
            # ax[0,1].imshow(grayscale_cam[0,0], vmax=1, vmin=0)
            # ax[0,2].imshow(grayscale_cam[0,1], vmax=1, vmin=0)
            # ax[0,3].imshow(grayscale_cam[0,2], vmax=1, vmin=0)
            # ax[0,4].imshow(grayscale_cam[0,3], vmax=1, vmin=0)
            # # ax[0,5].imshow(grayscale_cam[0,4], vmax=1, vmin=0)

            # ax[1,0].imshow(x[1].permute(1,2,0).cpu())
            # ax[1,1].imshow(grayscale_cam[1,0], vmax=1, vmin=0)
            # ax[1,2].imshow(grayscale_cam[1,1], vmax=1, vmin=0)
            # ax[1,3].imshow(grayscale_cam[1,2], vmax=1, vmin=0)
            # ax[1,4].imshow(grayscale_cam[1,3], vmax=1, vmin=0)
            # # ax[1,5].imshow(grayscale_cam[1,4], vmax=1, vmin=0)

            # ax[2,0].imshow(x[2].permute(1,2,0).cpu())
            # ax[2,1].imshow(grayscale_cam[2,0], vmax=1, vmin=0)
            # ax[2,2].imshow(grayscale_cam[2,1], vmax=1, vmin=0)
            # ax[2,3].imshow(grayscale_cam[2,2], vmax=1, vmin=0)
            # ax[2,4].imshow(grayscale_cam[2,3], vmax=1, vmin=0)
            # # ax[2,5].imshow(grayscale_cam[2,4], vmax=1, vmin=0)

            return grayscale_cam.detach()

        else:
            for image_num in range(self.batch_size):
                grayscale_cam = torch.zeros((self.batch_size, 1, x.size(2), x.size(3)))
                update_indicies = torch.where(i==image_num)[0].tolist()
                if update_indicies:
                    grayscale_cam[image_num] = cams[update_indicies].mean(dim=0)
            return grayscale_cam.cpu().detach().item()



        
#         for idx in testIndx:
#             class_nums = torch.where(org_class[idx]==1)[0].tolist()
#             targets=[]
#             # for class_num in class_nums: 
#             targets.append(Neg_Pos_ClassifierOutputTarget(category=class_nums, sign=toggle_sign[idx], device=self.device))
#                 # targets.append(SimilarityToConceptTarget(class_num))
#                 # targets.append(DifferenceFromConceptTarget(class_num))
#             grayscale_cam = cam(torch.unsqueeze(x[idx], dim=0), targets=targets)
#             fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
#             fig.suptitle('original image and cam attention')
#             ax1.imshow(x[idx].cpu().permute(1, 2, 0))
#             ax2.imshow(grayscale_cam[0])
#  # 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'
#             # ax2.imshow(cam(input_tensor=x[10].reshape(1,3,128,128), targets=[ClassifierOutputTarget(4)])[0])
#             # grayscale_cam = torch.tensor((grayscale_cam*2)-1)
#             # grayscale_cam = grayscale_cam.view(grayscale_cam.size(0), 1, grayscale_cam.size(1), grayscale_cam.size(2))
#         ############### end test case


        # toggled_classes = torch.where((org_class - desired_class)==1, 1, 0)
        # for indx, item in enumerate(toggled_classes):
        #     # number of classes that are toggled
        #     if torch.any(item):
        #         cam_of_each_class = []
        #         for class_num in torch.where(item==1)[0].tolist():
        #             targets = [ClassifierOutputTarget(class_num)]
        #             cam_of_each_class.append(cam(input_tensor= torch.unsqueeze(x[indx], dim=0), targets=targets))
        #         grayscale_cam.append(torch.mean(torch.tensor(cam_of_each_class), axis=0))
        #     else: 
        #         grayscale_cam.append(torch.zeros((1, x[indx].size(1), x[indx].size(2))))
            
            
        # targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1)]
        # grayscale_cam = cam(input_tensor=x)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('original image and cam attention')
        # ax1.imshow(x[15].cpu().permute(1, 2, 0))
        # ax2.imshow((grayscale_cam[15]*2)-1)
        # 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'
        # ax2.imshow(cam(input_tensor=x[10].reshape(1,3,128,128), targets=[ClassifierOutputTarget(4)])[0])
        # grayscale_cam = torch.tensor((grayscale_cam*2)-1)
        # grayscale_cam = grayscale_cam.view(grayscale_cam.size(0), 1, grayscale_cam.size(1), grayscale_cam.size(2))
        #perhaps it is better to return class_idx, it is needed to be tested ??????????????????  
        
    # def attention_generation(model, data_loader, save_attention_dir):
    #     decision_model = copy.deepcopy(model)
    #     # print(ADSModel)
    #     target_layers = [decision_model.feat_extract.feat_extract.features.denseblock4.denselayer16.conv2]
    #     cam = GradCAM(model=decision_model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    #     # targets = [ClassifierOutputTarget(281)]
    #     for batch_idx, batch_data in enumerate(data_loader):
    #             input_img = batch_data['image'].to(device)
    #             outputs = decision_model(input_img)
    #             # get the softmax probabilities
    #             probs = F.softmax(outputs).data.squeeze()
    #             # get the class indices of top k probabilities
    #             class_idx = topk(probs, 1)[1].int()
    #             for idx, tensorimage in enumerate(input_img):
                    
    #                 numpyImg = tensorimage.cpu().numpy().transpose(1, 2, 0)
    #                 img = cv2.cvtColor(numpyImg*255, cv2.COLOR_RGB2BGR)
    #                 cv2.imwrite(config.attention_map_dir+'/image/'+batch_data['name'][idx].split("/")[-1], img)

    #                 grayscale_cam = cam(input_tensor=tensorimage[None, ...])
    #                 # plt.imshow(grayscale_cam[0]*255, cmap='gray', vmin=0, vmax=255)
    #                 # plt.show()
    #                 grayscale_cam_img = cv2.cvtColor(grayscale_cam[0]*255, cv2.IMREAD_GRAYSCALE)
    #                 cv2.imwrite(config.attention_map_dir+'/grayscale_cam/'+batch_data['name'][idx].split("/")[-1], grayscale_cam_img)

    #                 visualization = show_cam_on_image(numpyImg, grayscale_cam[0], use_rgb=True)
    #                 cv2.imwrite(config.attention_map_dir+'/visualization/'+batch_data['name'][idx].split("/")[-1], visualization)

    #                 cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    #                 cv2.imwrite(config.attention_map_dir+'/cam_image/'+batch_data['name'][idx].split("/")[-1], cam_image)

    #                 # sample_path = os.path.join(save_attention_dir, '{}-attention.jpg'.format(i+1))
    #                 # save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    
    #     print('Saved decision model attention map into {}...'.format(save_attention_dir))


                    
# # # # attention map generation
#     if not os.path.exists(config.attention_map_dir):
#         os.mkdir(config.attention_map_dir)

#     if config.dataset in ['CelebA']:
#         DM_celeba_loader_train, DM_celeba_loader_val = decision_model_get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
#                                    config.celeba_crop_size, config.image_size, config.batch_size,
#                                    'CelebA', config.mode, config.num_workers, None, None)
#         decision_model = DecisionModel(opt=config, data_loader_train=DM_celeba_loader_train, 
#                                        data_loader_val=DM_celeba_loader_val, device=device)
#         decision_model.train() if (config.decision_model_train) else decision_model.evaluate_one_epoch()
#         attention_generation(decision_model.model, celeba_loader, config.attention_map_dir)
#     if config.dataset in ['RaFD']:
#         DM_rafd_loader_train, DM_rafd_loader_val = decision_model_get_loader(config.rafd_image_dir, None, None,
#                                  config.rafd_crop_size, config.image_size, config.batch_size,
#                                  'RaFD', config.mode, config.num_workers, None, None)
#         decision_model = DecisionModel(opt=config, data_loader_train=DM_rafd_loader_train,
#                                         data_loader_val=DM_rafd_loader_val, device=device)
#         decision_model.train() if (config.decision_model_train) else decision_model.evaluate_one_epoch()
#         attention_generation(decision_model.model, rafd_loader, config.attention_map_dir)
#     if config.dataset in ['BDD']:
#         DM_bdd_loader_train, DM_bdd_loader_val = decision_model_get_loader(image_dir=None, attr_path=None, selected_attrs=None,
#                                  crop_size=config.bdd_load_size, image_size=config.image_size, batch_size=config.batch_size,
#                                  dataset='BDD', mode=config.mode, num_workers=config.num_workers,
#                                  image_root=config.bdd_image_root, gt_root_train=config.bdd_gt_root_train)
#         decision_model = DecisionModel(opt=config, data_loader_train=DM_bdd_loader_train,
#                                         data_loader_val=DM_bdd_loader_val, device=device)
#         decision_model.train() if (config.decision_model_train) else decision_model.evaluate_one_epoch()
#         attention_generation(decision_model.model, bdd_loader, config.attention_map_dir)

# # visulize data
#     if config.dataset in ['CelebA']:
#         expImage = DataLoader_batch_sampler(celeba_loader)
#     if config.dataset in ['RaFD']:
#         expImage = DataLoader_batch_sampler(rafd_loader)
#     if config.dataset in ['BDD']:
#         expImage = DataLoader_batch_sampler(bdd_loader)
    
#     for idx, tensorimage in enumerate(expImage['image']):
#         numpyImg = tensorimage.cpu().numpy().transpose(1, 2, 0)
#         plt.imshow(numpyImg)
#         plt.show()