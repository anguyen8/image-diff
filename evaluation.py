import warnings
from argparse import ArgumentParser
import os
import json

import pytorch_lightning as pl
import torch
import numpy as np
from easydict import EasyDict
from loguru import logger as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from data.datamodule import DataModule
from models.centernet_with_coam import CenterNetWithCoAttention, suppress_non_maximum
from utils_cyws.general import get_easy_dict_from_yaml_file
from utils_cyws.matching_box import count_matching_boxes
from models.sfnnet import DeepEMD
from models.sfn_utils import load_model
from utils_cyws.post_process import change_box_detector, post_processing, matching_boxes
from modules.model import Model
from utils import create_batch_from_image_path, create_batch_from_image_data, fill_in_the_missing_information, prepare_batch_for_model
from modules.correspondence_extractor import CorrespondenceExtractor

from tqdm import tqdm
import time

from data.inpainted_coco_dataset import InpatinedCocoDataset
warnings.filterwarnings("ignore")

# Device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

@rank_zero_only
def print_args(configs):
    L.log("INFO", configs)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="centernet")
    parser.add_argument("--no_logging", action="store_true", default=True)
    parser.add_argument("--test_from_checkpoint", type=str, default="./checkpoints/pretrained-resnet50-3x-coam-scSE-affine.ckpt")
    parser.add_argument("--config_file", type=str, default='./configs/detection_resnet50_3x_coam_layers_affine.yml')
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--load_weights_from", type=str, default=None)
    parser.add_argument("--load_cyws_3d", type=str, default='./checkpoints/cyws-3d.ckpt')
    parser.add_argument("--transform", help="Type of Transform", type=str, default="single")
    parser.add_argument('--type', help="Type of coco inpainted dataset", type=str, default='large')
    parser.add_argument("--matching", help="Type of Matching method", type=str, default="unet_encoder_hungarian")
    parser.add_argument("--paper", help="Display for image in WACV paper", default=False)
    parser.add_argument("--wacv", help="Display as same WACV paper", default=False)
    parser.add_argument("--use_neighbor", help="Use neighbor information", default=False)
    parser.add_argument("--dst", help="Folder to save visualization result", default='./visualization_paper_results/wacv/')
    parser.add_argument("--left_to_right", default=True)
    parser.add_argument("--use_ground_truth", default=False)
    parser.add_argument("--num_sample", help="Number of sample for visualization", default=100)
    parser.add_argument("--data_name", help="Name of dataset", default="coco")
    parser.add_argument("--image_mode", help="Transform or not", default="transform")
    parser.add_argument("--top_n_bboxes", help="Select most confidence boxes", default=5)
    parser.add_argument("--use_nms", help="Use NMS to rememove duplicate boxes", default=False)
    parser.add_argument("--ratio", default=np.asarray([1.0, 1.0, 1.0, 1.0]))
    parser.add_argument("--iou_threshold", help="Threshold for remove boxes from left and right images", default=0.0)
    parser.add_argument("--iou_match_threshold", help="Threshold for accept two boxes in left and right images are corresponding", default=0.1)
    parser.add_argument("--remove_noise", help="Remove the noise bounding boxes", default=False)
    parser.add_argument("--visualize", help="Option to save the visualization after removing the noise boxes", default=False)
    parser.add_argument("--topk", help="Select top k predicted boxes with the most confidence scores", default=10)
    parser.add_argument("--change", default=True)
    parser.add_argument("--det_thres", help="Threshold for select predicted bounding boxes", default=0.1)
    parser.add_argument("--bbox_name", help="Name of dataset for storing bounding box", default="our_model_kubric")
    parser.add_argument("--iou_matching_counting", help="iou for counting matching", default=0.5)
    parser.add_argument("--use_encoder_feature", help="Choose to use encoder feature embedding. Without cost_matrix_method, default \
                        is use average feature map", default=False)
    parser.add_argument("--cost_matrix_method", help="Select method for calculate cost matrix", default='unet_encoder')
    # about model
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('-metric', type=str, default='cosine', choices=[ 'cosine' ])
    parser.add_argument('-norm', type=str, default='center', choices=[ 'center'])
    parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    #deepemd fcn only
    parser.add_argument('-feature_pyramid', type=str, default=None)
    #deepemd grid only patch_list
    parser.add_argument('-patch_list',type=str,default='2,3')
    parser.add_argument('-patch_ratio',type=float,default=2)
    # solver
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
    # others
    parser.add_argument('-emd_model_dir', type=str, default='./checkpoints/deepemd_trained_model/miniimagenet/fcn/max_acc.pth')
    args, _ = parser.parse_known_args()
    if args.method == "centernet":
        parser = CenterNetWithCoAttention.add_model_specific_args(parser)
    else:
        raise NotImplementedError(f"Unknown method type {args.method}")

    # parse configs from cmd line and config file into an EasyDict
    parser = DataModule.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args = EasyDict(vars(args))
    configs = get_easy_dict_from_yaml_file(args.config_file)

    # copy cmd line configs into config file configs, overwriting any duplicates
    for key in args.keys():
        if args[key] is not None:
            configs[key] = args[key]
        elif key not in configs.keys():
            configs[key] = args[key]

    print_args(configs)

    datamodule = DataModule(configs)
    test_dataloaders = datamodule.test_dataloader()


    # SFC (Structured Fully Connected) DeepEMD paper
    emd_model = DeepEMD(args)
    emd_model = load_model(emd_model, args.emd_model_dir)
    emd_model = emd_model.to(device)
    emd_model.eval()
    if args.feature_pyramid is not None:
        args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
    args.patch_list = [int(x) for x in args.patch_list.split(',')]

    # CYWS-3D model
    configs_cyws_3d = get_easy_dict_from_yaml_file("./config.yml")
    cyws_3d_model = Model(configs_cyws_3d, load_weights_from=args.load_cyws_3d).to(device)
    correspondence_extractor = CorrespondenceExtractor().to(device)
    depth_predictor = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).eval().to(device)

    pl.seed_everything(1, workers=True)
    datamodule = DataModule(configs)
    if configs.method == "centernet":
        diff_model = CenterNetWithCoAttention(configs)
        diff_model = diff_model.to(device)
        diff_model.eval()

    configs.gpus = 1

    multiple_test_set_outputs = []
    for dataloader_index, test_dataloader in enumerate(test_dataloaders):
        test_set_outputs = []
        for batch_index, batch in tqdm(enumerate(test_dataloader)):
            # Test with emd-hungarian method
            if args.matching == "emd_hungarian":
                predict = diff_model.test_step_emd(emd_model, batch, batch_index, dataloader_index)
                test_set_outputs.append(predict)
            if args.matching == "unet_encoder_hungarian":
                predict = diff_model.test_step_unet_encoder(emd_model, batch, batch_index, dataloader_index)
                test_set_outputs.append(predict)
            # Test with remove noise method
            elif args.matching == "post_processing":
                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last = change_box_detector(args, diff_model, batch)
                predict = post_processing(args, cyws_3d_model, depth_predictor, correspondence_extractor, batch["left_image"][0], batch["right_image"][0], 
                                          left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, None, "", use_path=False)
                test_set_outputs.append(predict)
            elif args.matching == "post_processing_and_emd_hungarian":
                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last = change_box_detector(args, diff_model, batch)
                predict = matching_boxes(args, cyws_3d_model, depth_predictor, correspondence_extractor, emd_model, batch["left_image"][0], batch["right_image"][0], 
                                          left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, None, None, None, None, use_path=False)
                test_set_outputs.append(predict)
            elif args.matching == "post_processing_and_encoder_hungarian":
                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last = change_box_detector(args, diff_model, batch)
                predict = matching_boxes(args, cyws_3d_model, depth_predictor, correspondence_extractor, diff_model, batch["left_image"][0], batch["right_image"][0], 
                                          left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                          left_image_encoded_features_last, right_image_encoded_features_last, None, None, use_path=False)
                test_set_outputs.append(predict)
            else:
                raise ValueError("Cannot find the suitable method")
            
        if len(test_dataloaders) == 1:
            multiple_test_set_outputs = test_set_outputs
        else:
            multiple_test_set_outputs.append(test_set_outputs)
    
    if args.change:
        # Test mAP and Recall
        diff_model.test_epoch_end(multiple_test_set_outputs)

        # Test maching number
        if len(diff_model.test_set_names) == 1:
            count_matching_boxes([multiple_test_set_outputs], iou_thresh=args.iou_matching_counting)
        else:
            count_matching_boxes(multiple_test_set_outputs, iou_thresh=args.iou_matching_counting)
    else:
        if len(diff_model.test_set_names) == 1:
            multiple_test_set_outputs = [multiple_test_set_outputs]

        for test_set_batch_outputs in multiple_test_set_outputs:
            num_match_pairs = 0
            for test_set_outputs in test_set_batch_outputs:
                (
                left_predicted_bboxes,
                right_predicted_bboxes,
                left_target_bboxes,
                right_target_bboxes,
                ) = test_set_outputs

                for bboxes_per_image in left_predicted_bboxes:
                    bboxes_per_image = bboxes_per_image[0][bboxes_per_image[1] == 0]
                    num_match_pairs += bboxes_per_image.shape[0]
            print('Num of matching boxes: {}'.format(num_match_pairs))
            print('Num of ground truth boxes: {}'.format(0))

            

