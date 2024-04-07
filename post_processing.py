import os
import numpy as np
from argparse import ArgumentParser
import kornia as K
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as label_connected_components
import shapely.geometry
from einops import rearrange
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import pil_to_tensor
from natsort import natsorted
from modules.model import Model
from utils import create_batch_from_image_path, create_batch_from_image_data, fill_in_the_missing_information, prepare_batch_for_model
from modules.correspondence_extractor import CorrespondenceExtractor

from data.datamodule import DataModule
from models.centernet_with_coam import CenterNetWithCoAttention
from utils_cyws.general import get_easy_dict_from_yaml_file
from utils_cyws.box_ops import center_distance
import torch
from models.sfnnet import DeepEMD
from models.sfn_utils import load_model
import torchvision.transforms as transforms
from utils_cyws.post_process import device, emd_transform, suppress_non_maximum, crop_image, emd_hungarian, cyws_3d, post_processing, \
    wacv, change_box_detector, matching_boxes, visualization_after_detection

from data.data_classes import COCOPair, STDPair, KubricPair, SynthtextPair

from utils_cyws.box_ops import box_iou

import warnings
warnings.filterwarnings('ignore')


def import_dataloader_collate_fn(method):
    if method == "centernet":
        from models.centernet_with_coam import dataloader_collate_fn
    else:
        raise NotImplementedError(f"Unknown method {method}")
    return dataloader_collate_fn


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = CenterNetWithCoAttention.add_model_specific_args(parser)
    parser = DataModule.add_data_specific_args(parser)
    parser.add_argument("--load_weights_from", type=str, default="./checkpoints/pretrained-resnet50-3x-coam-scSE-affine.ckpt")
    parser.add_argument("--load_cyws_3d", type=str, default='./checkpoints/cyws-3d.ckpt')
    parser.add_argument("--test_from_checkpoint", type=str, default="./checkpoints/pretrained-resnet50-3x-coam-scSE-affine.ckpt")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--transform", help="Type of Transform", type=str, default="single")
    parser.add_argument('--type', help="Type of coco inpainted dataset", type=str, default='large')
    parser.add_argument("--matching", help="Type of Matching method", type=str, default="matching_boxes")
    parser.add_argument("--paper", help="Display for image in WACV paper", default=False)
    parser.add_argument("--wacv", help="Display as same WACV paper", default=False)
    parser.add_argument("--use_neighbor", help="Use neighbor information", default=False)
    parser.add_argument("--dst", help="Folder to save visualization result",
                        default='./visualization_paper_results_pdf/no_change/wacv_random_select/kubric_post_processing_projective')
    parser.add_argument("--left_to_right", default=True)
    parser.add_argument("--use_ground_truth", default=False)
    parser.add_argument("--num_sample", help="Number of sample for visualization", default=500)
    parser.add_argument("--data_name", help="Name of dataset", default="coco")
    parser.add_argument("--image_mode", help="Transform or not", default="projective")
    parser.add_argument("--top_n_bboxes", help="Select most confidence boxes", default=100)
    parser.add_argument("--use_nms", help="Use NMS to rememove duplicate boxes", default=True)
    parser.add_argument("--ratio", default=np.asarray([1.0, 1.0, 1.0, 1.0]))
    parser.add_argument("--iou_threshold", help="Threshold for matching boxes tranform from left with boxes in the right image", default=0.0)
    parser.add_argument("--iou_match_threshold", help="Threshold for accept two boxes in left and right images are corresponding", default=0.1)
    parser.add_argument("--remove_noise", help="Remove the noise bounding boxes", default=True)
    parser.add_argument("--visualize", help="Option to save the visualization after removing the noise boxes", default=True)
    parser.add_argument("--topk", help="Select top k predicted boxes with the most confidence scores", default=10)
    parser.add_argument("--change", default=False)
    parser.add_argument("--image_transformation", default='projective')
    parser.add_argument("--det_thres", help="Threshold for select predicted bounding boxes", default=0.1)
    parser.add_argument("--use_encoder_feature", help="Choose to use encoder feature embedding. Without cost_matrix_method, default \
                        is use average feature map", default=False)
    parser.add_argument("--cost_matrix_method", help="Select method for calculate cost matrix", default='unet_encoder')
    parser.add_argument("--after_detect", help="Visualization after detection and set threshold", default=False)
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
    args = parser.parse_args()
    args = EasyDict(vars(args))
    configs = get_easy_dict_from_yaml_file(args.config_file)

    # copy cmd line configs into config file configs, overwriting any duplicates
    for key in args.keys():
        if args[key] is not None:
            configs[key] = args[key]
        elif key not in configs.keys():
            configs[key] = args[key]

    print(configs)

    # Test dataset
    if args.data_name == 'coco':
        dataset_path = '/home/hung/difference_localization/dataset/coco_inpainted/test/' + args.type
        test_images = natsorted(os.listdir(os.path.join('/home/hung/difference_localization/dataset/Test_Visualization/COCO-Inpainted/Images')))
        dataset = COCOPair(args=args, method="centernet", path_to_dataset=dataset_path, index=0, mode=args.image_mode)
        left_images = os.listdir(os.path.join(dataset_path, 'images_and_masks'))
        left_images = natsorted([left_image for left_image in left_images if 'mask' not in left_image])
        right_images = natsorted(os.listdir(os.path.join(dataset_path, 'inpainted')))
        image_pairs = [(left, right) for left, right in zip(left_images, right_images)]
        if args.paper:
            tmp_pairs = []
            for test_img in test_images:
                for img_pair in image_pairs:
                    if test_img == img_pair[0]:
                        tmp_pairs.append(img_pair)
            image_pairs = tmp_pairs
        else:
            random.seed(42)
            image_pairs = random.sample(image_pairs, args.num_sample)
    elif args.data_name == 'std':
        dataset_path = '/home/hung/difference_localization/dataset/std/'
        test_images = natsorted(os.listdir(os.path.join('/home/hung/difference_localization/dataset/Test_Visualization/STD')))
        dataset = STDPair(args=args, method="centernet", path_to_dataset=dataset_path, mode=args.image_mode)
        image_ids = dataset.image_ids
        if args.paper:
            test_images = natsorted(os.listdir(os.path.join('/home/hung/difference_localization/dataset/Test_Visualization/STD')))
            image_ids = [image.split('.')[0] for image in test_images if '_2' not in image]
        else:
            random.seed(42)
            image_ids = random.sample(image_ids, args.num_sample)
    elif args.data_name == 'kubric':
        dataset_path = '/home/hung/difference_localization/dataset/kubric-change/'
        test_images = natsorted(os.listdir(os.path.join('/home/hung/difference_localization/dataset/Test_Visualization/Kubric')))
        dataset = KubricPair(args=args, method="centernet", path_to_dataset=dataset_path, mode=args.image_mode)
        image_data = dataset.data.tolist()
        if args.paper:
            test_images = natsorted(os.listdir(os.path.join('/home/hung/difference_localization/dataset/Test_Visualization/Kubric')))
            image_data = [img_data for img_data in image_data if img_data[0].split('/')[-1] in test_images]
        else:
            random.seed(42)
            image_data = random.sample(image_data, args.num_sample)
    elif args.data_name == 'synthtext':
        dataset_path = '/home/hung/difference_localization/dataset/synthtext_change/'
        test_images = natsorted(os.listdir(os.path.join('/home/hung/difference_localization/dataset/Test_Visualization/Synthtext')))
        dataset = SynthtextPair(args=args, method="centernet", path_to_dataset=dataset_path, item_index=0, mode=args.image_mode)
        synthetic_image_names = dataset.synthetic_image_names
        image_ids = np.arange(1, 1000)
        if args.paper:
            image_ids = [synthetic_image_names.index(test_img + '_0') for test_img in test_images]
        else:
            random.seed(42)
            image_ids = random.sample(image_ids.tolist(), args.num_sample)
    else:
        raise ValueError("Cannot find the suitable dataset")

    # Difference model initialization (WACV paper)
    diff_model = CenterNetWithCoAttention(configs)
    diff_model = diff_model.to(device)
    diff_model.eval()

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

    # Extract bounding boxes
    if args.data_name == 'coco':
        for pair in tqdm(image_pairs):
            dataset = COCOPair(args, method="centernet", path_to_dataset=dataset_path,
                            index=pair[0].split('.')[0], mode=args.image_mode)
            dataset.path_to_image1 = os.path.join(dataset_path, 'images_and_masks', pair[0])
            if args.change:
                dataset.path_to_image2 = os.path.join(dataset_path, 'inpainted', pair[1])
            else:
                dataset.path_to_image2 = os.path.join(dataset_path, 'images_and_masks', pair[0])
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=import_dataloader_collate_fn("centernet"),
            )

            figure, subplots = plt.subplots(1, 2)
            for batch_with_single_item in dataloader:
                if batch_with_single_item is None:
                    continue
                subplots[0].imshow(K.tensor_to_image(batch_with_single_item["left_image"][0]))
                subplots[1].imshow(K.tensor_to_image(batch_with_single_item["right_image"][0]))

                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last = change_box_detector(args, diff_model, batch_with_single_item)
                if args.after_detect:
                    visualization_after_detection(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, pair[0])
                    continue

                if args.wacv:
                    wacv(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, pair[0])
                else:
                    left_predicted_bboxes = [bbox for bbox in left_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                    right_predicted_bboxes = [bbox for bbox in right_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]

                    if args.matching == 'emd_hungarian':
                        emd_hungarian(args, emd_model, left_predicted_bboxes, right_predicted_bboxes, 
                                      left_target_bboxes, right_target_bboxes, 
                                      batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0], subplots, pair[0], use_path=False)
                    elif args.matching == 'cyws_3d':
                        cyws_3d(args, cyws_3d_model, depth_predictor, correspondence_extractor, batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0],
                                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes,
                                subplots, pair[0], use_path=False)
                    elif args.matching == 'post_processing':
                        post_processing(args, cyws_3d_model, depth_predictor, correspondence_extractor, batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0],
                                        left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                        subplots, pair[0], use_path=False)
                    elif args.matching == 'matching_boxes':
                        matching_boxes(args, cyws_3d_model, depth_predictor, correspondence_extractor, emd_model, batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0], 
                                       left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                       left_image_encoded_features_last, right_image_encoded_features_last, subplots, pair[0], use_path=False)
                    else:
                        raise ValueError("Cannot find the suitable matching methods. Please check the parameter carefully")
    elif args.data_name == 'kubric':
        for img_data in tqdm(image_data):
            name = img_data[0].split('/')[-1]
            dataset.path_to_image1 = img_data[0]
            dataset.path_to_image2 = img_data[1]
            dataset.mask1_path = img_data[2]
            dataset.mask2_path = img_data[3]
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=import_dataloader_collate_fn("centernet"),
            )

            figure, subplots = plt.subplots(1, 2)
            for batch_with_single_item in dataloader:
                if batch_with_single_item is None:
                    continue
                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last = change_box_detector(args, diff_model, batch_with_single_item)

                subplots[0].imshow(K.tensor_to_image(batch_with_single_item["left_image"][0]))
                subplots[1].imshow(K.tensor_to_image(batch_with_single_item["right_image"][0]))

                if args.after_detect:
                    visualization_after_detection(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, name)
                    continue

                if args.wacv:
                    wacv(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, name)
                else:
                    left_predicted_bboxes = [bbox for bbox in left_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                    right_predicted_bboxes = [bbox for bbox in right_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                    if args.matching == 'emd_hungarian':
                        emd_hungarian(args, emd_model, left_predicted_bboxes, right_predicted_bboxes, 
                                      left_target_bboxes, right_target_bboxes, 
                                      dataset.path_to_image1, dataset.path_to_image2, subplots, name, use_path=True)
                    elif args.matching == 'cyws_3d':
                        cyws_3d(args, cyws_3d_model, depth_predictor, correspondence_extractor, dataset.path_to_image1, dataset.path_to_image2,
                                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes,
                                subplots, name, use_path=True)
                    elif args.matching == 'post_processing':
                        post_processing(args, cyws_3d_model, depth_predictor, correspondence_extractor, dataset.path_to_image1, dataset.path_to_image2,
                                        left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                        subplots, name, use_path=True)
                    elif args.matching == 'matching_boxes':
                        matching_boxes(args, cyws_3d_model, depth_predictor, correspondence_extractor, emd_model, batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0], 
                                       left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                       left_image_encoded_features_last, right_image_encoded_features_last, subplots, name, use_path=False)
                    else:
                        raise ValueError("Cannot find the suitable matching methods. Please check the parameter carefully")
    elif args.data_name == 'std':
        for image_id in tqdm(image_ids):
            dataset.path_to_image1 = os.path.join(dataset_path, f"{image_id}.png")
            dataset.path_to_image2 = os.path.join(dataset_path, f"{image_id}_2.png")
            dataset.image_id = image_id
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=import_dataloader_collate_fn("centernet"),
            )

            figure, subplots = plt.subplots(1, 2)
            for batch_with_single_item in dataloader:
                if batch_with_single_item is None:
                    continue
                
                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last = change_box_detector(args, diff_model, batch_with_single_item)

                subplots[0].imshow(K.tensor_to_image(batch_with_single_item["left_image"][0]))
                subplots[1].imshow(K.tensor_to_image(batch_with_single_item["right_image"][0]))

                if args.after_detect:
                    visualization_after_detection(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, image_id + '.png')
                    continue

                if args.wacv:
                    wacv(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, image_id + '.png')
                else:
                    left_predicted_bboxes = [bbox for bbox in left_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                    right_predicted_bboxes = [bbox for bbox in right_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]

                    if args.matching == 'emd_hungarian':
                        emd_hungarian(args, emd_model, left_predicted_bboxes, right_predicted_bboxes, 
                                      left_target_bboxes, right_target_bboxes, 
                                      dataset.path_to_image1, dataset.path_to_image2, subplots, image_id + '.png', use_path=True)
                    elif args.matching == 'cyws_3d':
                        cyws_3d(args, cyws_3d_model, depth_predictor, correspondence_extractor, dataset.path_to_image1, dataset.path_to_image2,
                                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes,
                                subplots, image_id + '.png', use_path=True)
                    elif args.matching == 'post_processing':
                        post_processing(args, cyws_3d_model, depth_predictor, correspondence_extractor, dataset.path_to_image1, dataset.path_to_image2,
                                        left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                        subplots, image_id + '.png', use_path=True)    
                    elif args.matching == 'matching_boxes':
                        matching_boxes(args, cyws_3d_model, depth_predictor, correspondence_extractor, emd_model, batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0], 
                                       left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                       left_image_encoded_features_last, right_image_encoded_features_last, subplots, image_id + '.png', use_path=False) 
                    else:
                        raise ValueError("Cannot find the suitable matching methods. Please check the parameter carefully")
    elif args.data_name == 'synthtext':
        for image_id in tqdm(image_ids):
            dataset = SynthtextPair(args=args, method="centernet", path_to_dataset=dataset_path, item_index=image_id, mode='identity')
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=import_dataloader_collate_fn("centernet"),
            )

            figure, subplots = plt.subplots(1, 2)
            for batch_with_single_item in dataloader:
                if batch_with_single_item is None:
                    continue
  
                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last = change_box_detector(args, diff_model, batch_with_single_item)

                subplots[0].imshow(K.tensor_to_image(batch_with_single_item["left_image"][0]))
                subplots[1].imshow(K.tensor_to_image(batch_with_single_item["right_image"][0]))

                if args.after_detect:
                    visualization_after_detection(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, dataset.original_image_names[image_id])
                    continue

                if args.wacv:
                    wacv(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, dataset.original_image_names[image_id])
                else:
                    left_predicted_bboxes = [bbox for bbox in left_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                    right_predicted_bboxes = [bbox for bbox in right_predicted_bboxes
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]

                    if args.matching == 'emd_hungarian':
                        left_img = Image.fromarray(np.uint8(K.tensor_to_image(batch_with_single_item["left_image"][0])*255)).convert('RGB')
                        right_img = Image.fromarray(np.uint8(K.tensor_to_image(batch_with_single_item["right_image"][0])*255)).convert('RGB')
                        emd_hungarian(args, emd_model, left_predicted_bboxes, right_predicted_bboxes, 
                                      left_target_bboxes, right_target_bboxes, 
                                      left_img, right_img, subplots, dataset.original_image_names[image_id], use_path=False)
                    elif args.matching == 'cyws_3d':
                        cyws_3d(args, cyws_3d_model, depth_predictor, correspondence_extractor, batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0],
                                left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes,
                                subplots, dataset.original_image_names[image_id], use_path=False)
                    elif args.matching == 'post_processing':
                        post_processing(args, cyws_3d_model, depth_predictor, correspondence_extractor, dataset.path_to_image1, dataset.path_to_image2,
                                        left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                        subplots, dataset.original_image_names[image_id], use_path=True) 
                    elif args.matching == 'matching_boxes':
                        matching_boxes(args, cyws_3d_model, depth_predictor, correspondence_extractor, emd_model, batch_with_single_item["left_image"][0], batch_with_single_item["right_image"][0], 
                                       left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                                       left_image_encoded_features_last, right_image_encoded_features_last, subplots, dataset.original_image_names[image_id], use_path=False) 
                    else:
                        raise ValueError("Cannot find the suitable matching methods. Please check the parameter carefully")







                        




