import argparse  
import os
import yaml
import pprint

def str2bool(v):
    return v.lower() in ('true')

def read_yaml(fileName):
    """ A function to read YAML file"""
    with open(fileName+'.yml') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config

def write_yaml(data,fileName):
    """ A function to write YAML file"""
    with open(fileName+'.yml', 'w') as f:
        yaml.dump(data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=4, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec_x', type=float, default=10, help='weight for reconstruction loss of image')
    parser.add_argument('--lambda_rec_sal', type=float, default=10, help='weight for reconstruction loss of saliency')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    # G of GAN model
    parser.add_argument('--g_loss_cls_of_d', type=bool, default=True, help='generator sees class prediction of D or decision model, true for D')
    parser.add_argument('--g_loss_sal_rec_method', type=str, default='mean', choices=['mean', 'bce'], help='loss between reconstructed saliency and real one')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both', 'BDD'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')#for bdd could be 8
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)#for bdd could be 4
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='dataset/celeba/images')
    parser.add_argument('--attr_path', type=str, default='dataset/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='sailencyGAN3_celeba/logs')
    parser.add_argument('--model_save_dir', type=str, default='sailencyGAN3_celeba/models')
    parser.add_argument('--sample_dir', type=str, default='sailencyGAN3_celeba/samples')
    parser.add_argument('--result_dir', type=str, default='sailencyGAN3_celeba/results')
    parser.add_argument('--config_dir', type=str, default='sailencyGAN3_celeba/')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    #decision model
    parser.add_argument('--decision_model_optimizer', type=str, default='adam')
    parser.add_argument('--decision_model_LOG_DIR', type=str, default='checkpoints/decision_densenet')
    parser.add_argument('--decision_model_attributes_idx', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--decision_model_lr', type=float, default=0.0001)
    parser.add_argument('--decision_model_num_epochs', type=int, default=4)
    parser.add_argument('--decisionModel', type=int, default=1000)
    parser.add_argument('--decision_model_name', type=str, default='decision_model_CelebA')
    parser.add_argument('--decision_model_checkpoints_dir', type=str, default='checkpoints/decision_densenet')
    parser.add_argument('--decision_model_train', type=str, default='', choices=['train', 'test', ''])
    parser.add_argument('--decision_Model_batch_size', type=int, default=16)
    parser.add_argument('--decision_Model_num_workers', type=int, default=4)#for bdd could be 4
        
    #for bdd loader
    parser.add_argument('--bdd_load_size', type=int, default=(512, 256))
    parser.add_argument('--bdd_data_dir', type=str, default="dataset/BDDIO_lastframe/")
    parser.add_argument('--bdd_image_root', type=str, default=os.path.join(parser.parse_args().bdd_data_dir, "data"))
    parser.add_argument('--bdd_gt_root_train', type=str, default=os.path.join(parser.parse_args().bdd_data_dir, "train_25k_images_actions.json"))
    parser.add_argument('--bdd_gt_root_val', type=str, default=os.path.join(parser.parse_args().bdd_data_dir, "val_25k_images_actions.json"))
        
    #attention map
    parser.add_argument('--attention_map_dir', type=str, default=os.path.join('dataset/attention_map/',parser.parse_args().dataset))
    
    # saliency_method
    parser.add_argument('--saliency_method', type=str, default='GradCAM', help=['AblationCAM', 'GradCAM', 'HiResCAM'])

    # metrics
    parser.add_argument('--metrics', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['FID', 'IS', 'sparcity', 'validity'])
    parser.add_argument('--metrics_dir', type=str, default='metrics/')
    parser.add_argument('--metrics_dir', type=int, default=5000)
    
    
    parser.add_argument('--wandb', type=bool, default=False)

    
    


    
    #counterfactual parameters
    #CF_mehtods: 
        #saliencyGAN uses attGAN with an extra generation of decisionModel's saliency
        #saliencyGAN_v2 uses starGAN with an extra generation of decisionModel's saliency  
        #saliencyGAN_v3 uses attGAN with an extra generation of decisionModel's saliency for each label class (c_dim)  
    parser.add_argument('--CF_method', type=str, default='sailencyGAN_v3', help=['attGAN', 'sailencyGAN', 'starGAN', 'sailencyGAN_v2', 'sailencyGAN_v3'])
    
    config = parser.parse_args()
    write_yaml(config, config.CF_method+'-'+config.dataset+'-'+config.saliency_method+'-config')
    
    my_config = read_yaml(config.CF_method+'-'+config.dataset+'-'+config.saliency_method+'-config')
    pprint.pprint(my_config)