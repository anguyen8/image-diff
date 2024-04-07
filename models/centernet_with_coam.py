import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from detectron2.structures.image_list import ImageList
from easydict import EasyDict
from loguru import logger as L
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from pytorch_lightning.utilities import rank_zero_only
from PIL import Image
import shapely.geometry
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import PrecisionRecallDisplay
import math
from einops import rearrange, repeat

import utils_cyws.general
import wandb
from data.datamodule import DataModule
from models.coattention import CoAttentionModule
from models.centernet_utils import get_bboxes
from segmentation_models_pytorch.unet.model import Unet
from utils_cyws.voc_eval import BoxList, eval_detection_voc
from utils_cyws import box_ops
from utils_cyws.box_ops import box_iou
from .sfnnet import DeepEMD
from .sfn_utils import load_model
from .matcher import Matcher
from argparse import ArgumentParser


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_batch_from_image_data, fill_in_the_missing_information, prepare_batch_for_model

plt.ioff()

# Device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# EMD transform
image_size = 84
emd_transform = transforms.Compose([
    transforms.Resize([92, 92]),
    transforms.CenterCrop(image_size),

    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

faster_rcnn_transform = transforms.Compose([transforms.ToTensor()])

# Extending bounding box ratio
ratio = np.asarray([1.0, 1.0, 1.0, 1.0])

# Topk bounding box
top_n_bboxes = 100

cyws_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


class CenterNetWithCoAttention(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.test_set_names = [test_set.name for test_set in args.datasets.test_datasets]
        self.lr = args.lr
        self.args = args
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        number_of_coam_layers, coam_input_channels, coam_hidden_channels = args.coam_layer_data
        self.unet_model = Unet(
            args.encoder,
            decoder_channels=(256, 256, 128, 128, 64),
            encoder_depth=5,
            encoder_weights="imagenet",
            num_coam_layers=number_of_coam_layers,
            decoder_attention_type=args.decoder_attention,
            disable_segmentation_head=True,
        )
        self.coattention_modules = nn.ModuleList(
            [
                CoAttentionModule(
                    input_channels=coam_input_channels[i],
                    hidden_channels=coam_hidden_channels[i],
                    attention_type=args.attention,
                )
                for i in range(number_of_coam_layers)
            ]
        )

        self.centernet_head = CenterNetHead(
            in_channel=64,
            feat_channel=64,
            num_classes=1,
            test_cfg=EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100}),
        )
        self.centernet_head.init_weights()
        if args.load_weights_from is not None:
            self.safely_load_state_dict(torch.load(args.load_weights_from, map_location=device)["state_dict"])

        self.matcher = Matcher(cost_bbox=5, cost_giou=2)
        # self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_matching = nn.Linear(args.inchannels, args.outchannels)
        self.test_cfg = EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100, "thres": 0.1})
        # self.test_cfg = EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100})

    def safely_load_state_dict(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in checkpoint_state_dict:
            if k in model_state_dict:
                if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                    L.log(
                        "INFO",
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {checkpoint_state_dict[k].shape}",
                    )
                    checkpoint_state_dict[k] = model_state_dict[k]
            else:
                L.log("INFO", f"Dropping parameter {k}")
        self.load_state_dict(checkpoint_state_dict, strict=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CenterNetWithCoAttention")
        parser.add_argument("--lr", type=float)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--encoder", type=str, choices=["resnet50", "resnet18"])
        parser.add_argument("--coam_layer_data", nargs="+", type=int)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--decoder_attention", type=str, default="scse")

        return parent_parser
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        src_boxes = torch.cat([s[i] for s, (i, _) in zip(outputs, indices)], dim=0)
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes = src_boxes/256.0
        target_boxes = target_boxes/256.0

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_match(self, left_image, right_image, left_predicted_bboxes, right_predicted_bboxes, left_indices, right_indices):
        left_image = Image.fromarray(np.uint8(K.tensor_to_image(right_image)*255)).convert('RGB')
        right_image = Image.fromarray(np.uint8(K.tensor_to_image(right_image)*255)).convert('RGB')
        left_predict_boxes = torch.cat([s[i] for s, (i, _) in zip(left_predicted_bboxes.unsqueeze(0), np.expand_dims(left_indices, 0))], dim=0)
        right_predict_boxes = torch.cat([s[i] for s, (i, _) in zip(right_predicted_bboxes.unsqueeze(0), np.expand_dims(right_indices, 0))], dim=0)

        left_predicted_features = torch.cat([torch.flatten(self.avg(self.encode(cyws_transform(left_image.crop(bbox[:4]*ratio)).unsqueeze(0).to(device))), 1) for bbox in left_predict_boxes.detach().cpu().numpy()], dim=0)
        right_predicted_features = torch.cat([torch.flatten(self.avg(self.encode(cyws_transform(right_image.crop(bbox[:4]*ratio)).unsqueeze(0).to(device))), 1) for bbox in right_predict_boxes.detach().cpu().numpy()], dim=0)

        similarity_map = 1.0 - self.cosine_similarity(left_predicted_features, right_predicted_features).detach().cpu().numpy()
        indices = linear_sum_assignment(similarity_map)

        left_predicted_features = left_predicted_features[indices[0]]
        right_predicted_features = right_predicted_features[indices[1]]
        features = torch.cat([left_predicted_features, right_predicted_features], dim=1)
        features = self.fc_matching(features)
        target = torch.from_numpy(indices[0] == indices[1]).long().to(device)

        loss = F.cross_entropy(features, target)
        return loss
    
    def loss_match_with_context(self, left_predicted_bboxes, right_predicted_bboxes, left_indices, right_indices, left_image_encoded_features, right_image_encoded_features):
        left_predict_boxes = torch.cat([s[i] for s, (i, _) in zip(left_predicted_bboxes.unsqueeze(0), np.expand_dims(left_indices, 0))], dim=0)
        right_predict_boxes = torch.cat([s[i] for s, (i, _) in zip(right_predicted_bboxes.unsqueeze(0), np.expand_dims(right_indices, 0))], dim=0)  

        left_predicted_features = []
        right_predicted_features = []
        if len(left_image_encoded_features) == 1:
            left_image_encoded_features = torch.flatten(left_image_encoded_features[0].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
            right_image_encoded_features = torch.flatten(right_image_encoded_features[0].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
            left_predict_index = torch.floor(left_predict_boxes/32)
            right_predict_index = torch.floor(right_predict_boxes/32)

            for idx in range(len(left_predict_index)):
                x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), 7), min(max(int(left_predict_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), 7), min(max(int(left_predict_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 1]*8 + coor[:, 0], dtype=torch.long)
                left_predicted_features.append(torch.mean(left_image_encoded_features[:, flatten_coor, :], dim=1))
            left_predicted_features = torch.cat(left_predicted_features, dim=0)

            for idx in range(len(right_predict_index)):
                x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), 7), min(max(int(right_predict_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), 7), min(max(int(right_predict_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 0]*8 + coor[:, 1], dtype=torch.long)
                right_predicted_features.append(torch.mean(right_image_encoded_features[:, flatten_coor, :], dim=1))
            right_predicted_features = torch.cat(right_predicted_features, dim=0)
        else:
            for i in range(len(left_image_encoded_features)):
                left_image_encoded_features_tmp = torch.flatten(left_image_encoded_features[i].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
                right_image_encoded_features_tmp = torch.flatten(right_image_encoded_features[i].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
                left_predict_index = torch.floor(left_predict_boxes/math.pow(2, 3+i))
                right_predict_index = torch.floor(right_predict_boxes/math.pow(2, 3+i))
                
                left_predicted_features_tmp = []
                right_predicted_features_tmp = []
                for idx in range(len(left_predict_index)):
                    x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_predict_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_predict_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 1]*int(math.pow(2, 5-i)) + coor[:, 0], dtype=torch.long)
                    left_predicted_features_tmp.append(torch.mean(left_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                left_predicted_features_tmp = torch.cat(left_predicted_features_tmp, dim=0)
                left_predicted_features.append(left_predicted_features_tmp)

                for idx in range(len(right_predict_index)):
                    x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_predict_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_predict_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 0]*int(math.pow(2, 5-i)) + coor[:, 1], dtype=torch.long)
                    right_predicted_features_tmp.append(torch.mean(right_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                right_predicted_features_tmp = torch.cat(right_predicted_features_tmp, dim=0)
                right_predicted_features.append(right_predicted_features_tmp)

            left_predicted_features = torch.cat(left_predicted_features, dim=1)
            right_predicted_features = torch.cat(right_predicted_features, dim=1)

        similarity_map = 1.0 - self.cosine_similarity(left_predicted_features, right_predicted_features).detach().cpu().numpy()
        indices = linear_sum_assignment(similarity_map)

        left_predicted_features = left_predicted_features[indices[0]]
        right_predicted_features = right_predicted_features[indices[1]]
        features = torch.cat([left_predicted_features, right_predicted_features], dim=1)
        features = self.fc_matching(features)
        target = torch.from_numpy(indices[0] == indices[1]).long().to(device)

        loss = F.cross_entropy(features, target)
        return loss

    def loss_match_with_context_ground_truth(self, left_target_bboxes, right_target_bboxes, left_image_encoded_features, right_image_encoded_features):
        left_predicted_features = []
        right_predicted_features = []

        if len(left_image_encoded_features) == 1:
            left_image_encoded_features = torch.flatten(left_image_encoded_features[0].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
            right_image_encoded_features = torch.flatten(right_image_encoded_features[0].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
            left_predict_index = torch.floor(torch.tensor(left_target_bboxes)/32).to(device)
            right_predict_index = torch.floor(torch.tensor(right_target_bboxes)/32).to(device)
            for idx in range(len(left_predict_index)):
                x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), 7), min(max(int(left_predict_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), 7), min(max(int(left_predict_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 1]*8 + coor[:, 0], dtype=torch.long)
                left_predicted_features.append(torch.mean(left_image_encoded_features[:, flatten_coor, :], dim=1))
            left_predicted_features = torch.cat(left_predicted_features, dim=0)

            for idx in range(len(right_predict_index)):
                x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), 7), min(max(int(right_predict_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), 7), min(max(int(right_predict_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 0]*8 + coor[:, 1], dtype=torch.long)
                right_predicted_features.append(torch.mean(right_image_encoded_features[:, flatten_coor, :], dim=1))
            right_predicted_features = torch.cat(right_predicted_features, dim=0)
        else:
            for i in range(len(left_image_encoded_features)):
                left_image_encoded_features_tmp = torch.flatten(left_image_encoded_features[i].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
                right_image_encoded_features_tmp = torch.flatten(right_image_encoded_features[i].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
                left_predict_index = torch.floor(torch.tensor(left_target_bboxes)/math.pow(2, 3+i)).to(device)
                right_predict_index = torch.floor(torch.tensor(right_target_bboxes)/math.pow(2, 3+i)).to(device)
                
                left_predicted_features_tmp = []
                right_predicted_features_tmp = []
                for idx in range(len(left_predict_index)):
                    x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_predict_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_predict_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 1]*int(math.pow(2, 5-i)) + coor[:, 0], dtype=torch.long)
                    left_predicted_features_tmp.append(torch.mean(left_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                left_predicted_features_tmp = torch.cat(left_predicted_features_tmp, dim=0)
                left_predicted_features.append(left_predicted_features_tmp)

                for idx in range(len(right_predict_index)):
                    x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_predict_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_predict_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 0]*int(math.pow(2, 5-i)) + coor[:, 1], dtype=torch.long)
                    right_predicted_features_tmp.append(torch.mean(right_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                right_predicted_features_tmp = torch.cat(right_predicted_features_tmp, dim=0)
                right_predicted_features.append(right_predicted_features_tmp)

            left_predicted_features = torch.cat(left_predicted_features, dim=1)
            right_predicted_features = torch.cat(right_predicted_features, dim=1)

        similarity_map = 1.0 - self.cosine_similarity(left_predicted_features, right_predicted_features).detach().cpu().numpy()
        indices = linear_sum_assignment(similarity_map)

        left_predicted_features = left_predicted_features[indices[0]]
        right_predicted_features = right_predicted_features[indices[1]]
        features = torch.cat([left_predicted_features, right_predicted_features], dim=1)
        features = self.fc_matching(features)
        target = torch.from_numpy(indices[0] == indices[1]).long().to(device)
        # target = torch.ones(features.shape[0]).long().to(device)

        loss = F.cross_entropy(features, target)
        return loss
    
    def loss_match_with_context_mix_ground_truth(self, left_predicted_bboxes, right_predicted_bboxes, left_indices, right_indices,
                                                 left_target_bboxes, right_target_bboxes, left_image_encoded_features, right_image_encoded_features):
        left_predict_boxes = torch.cat([s[i] for s, (i, _) in zip(left_predicted_bboxes.unsqueeze(0), np.expand_dims(left_indices, 0))], dim=0)
        right_predict_boxes = torch.cat([s[i] for s, (i, _) in zip(right_predicted_bboxes.unsqueeze(0), np.expand_dims(right_indices, 0))], dim=0)

        left_predicted_features = []
        right_predicted_features = []
        left_target_features = []
        right_target_features = []
        if len(left_image_encoded_features) == 1:
            left_image_encoded_features = torch.flatten(left_image_encoded_features[0].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
            right_image_encoded_features = torch.flatten(right_image_encoded_features[0].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)

            left_predict_index = torch.floor(left_predict_boxes/32)
            right_predict_index = torch.floor(right_predict_boxes/32)
            left_target_index = torch.floor(torch.tensor(left_target_bboxes)/32).to(device)
            right_target_index = torch.floor(torch.tensor(right_target_bboxes)/32).to(device)

            # Feature for predicted boxes
            for idx in range(len(left_predict_index)):
                x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), 7), min(max(int(left_predict_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), 7), min(max(int(left_predict_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 1]*8 + coor[:, 0], dtype=torch.long)
                left_predicted_features.append(torch.mean(left_image_encoded_features[:, flatten_coor, :], dim=1))
            left_predicted_features = torch.cat(left_predicted_features, dim=0)

            for idx in range(len(right_predict_index)):
                x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), 7), min(max(int(right_predict_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), 7), min(max(int(right_predict_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 0]*8 + coor[:, 1], dtype=torch.long)
                right_predicted_features.append(torch.mean(right_image_encoded_features[:, flatten_coor, :], dim=1))
            right_predicted_features = torch.cat(right_predicted_features, dim=0)

            # Feature for target boxes
            for idx in range(len(left_target_index)):
                x_coor = torch.arange(min(max(int(left_target_index[idx][0]), 0), 7), min(max(int(left_target_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(left_target_index[idx][1]), 0), 7), min(max(int(left_target_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 1]*8 + coor[:, 0], dtype=torch.long)
                left_target_features.append(torch.mean(left_image_encoded_features[:, flatten_coor, :], dim=1))
            left_target_features = torch.cat(left_target_features, dim=0)

            for idx in range(len(right_target_index)):
                x_coor = torch.arange(min(max(int(right_target_index[idx][0]), 0), 7), min(max(int(right_target_index[idx][2]), 0), 7) + 1)
                y_coor = torch.arange(min(max(int(right_target_index[idx][1]), 0), 7), min(max(int(right_target_index[idx][3]), 0), 7) + 1)
                coor = torch.cartesian_prod(x_coor, y_coor)
                flatten_coor = torch.tensor(coor[:, 0]*8 + coor[:, 1], dtype=torch.long)
                right_target_features.append(torch.mean(right_image_encoded_features[:, flatten_coor, :], dim=1))
            right_target_features = torch.cat(right_target_features, dim=0)
        else:
            for i in range(len(left_image_encoded_features)):
                left_image_encoded_features_tmp = torch.flatten(left_image_encoded_features[i].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)
                right_image_encoded_features_tmp = torch.flatten(right_image_encoded_features[i].permute(1, 2, 0), start_dim=0, end_dim=1).unsqueeze(0)

                left_predict_index = torch.floor(left_predict_boxes/math.pow(2, 3+i))
                right_predict_index = torch.floor(right_predict_boxes/math.pow(2, 3+i))
                left_target_index = torch.floor(torch.tensor(left_target_bboxes)/math.pow(2, 3+i)).to(device)
                right_target_index = torch.floor(torch.tensor(right_target_bboxes)/math.pow(2, 3+i)).to(device)

                left_predicted_features_tmp = []
                right_predicted_features_tmp = []
                left_target_features_tmp = []
                right_target_features_tmp = []

                # Feature for predicted boxes
                for idx in range(len(left_predict_index)):
                    x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_predict_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_predict_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 1]*int(math.pow(2, 5-i)) + coor[:, 0], dtype=torch.long)
                    left_predicted_features_tmp.append(torch.mean(left_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                left_predicted_features_tmp = torch.cat(left_predicted_features_tmp, dim=0)
                left_predicted_features.append(left_predicted_features_tmp)

                for idx in range(len(right_predict_index)):
                    x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_predict_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_predict_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 0]*int(math.pow(2, 5-i)) + coor[:, 1], dtype=torch.long)
                    right_predicted_features_tmp.append(torch.mean(right_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                right_predicted_features_tmp = torch.cat(right_predicted_features_tmp, dim=0)
                right_predicted_features.append(right_predicted_features_tmp)

                # Feature for target boxes
                for idx in range(len(left_target_index)):
                    x_coor = torch.arange(min(max(int(left_target_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_target_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(left_target_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(left_target_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 1]*int(math.pow(2, 5-i)) + coor[:, 0], dtype=torch.long)
                    left_target_features_tmp.append(torch.mean(left_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                left_target_features_tmp = torch.cat(left_target_features_tmp, dim=0)
                left_target_features.append(left_target_features_tmp)

                for idx in range(len(right_target_index)):
                    x_coor = torch.arange(min(max(int(right_target_index[idx][0]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_target_index[idx][2]), 0), int(math.pow(2, 5-i))-1) + 1)
                    y_coor = torch.arange(min(max(int(right_target_index[idx][1]), 0), int(math.pow(2, 5-i))-1), min(max(int(right_target_index[idx][3]), 0), int(math.pow(2, 5-i))-1) + 1)
                    coor = torch.cartesian_prod(x_coor, y_coor)
                    flatten_coor = torch.tensor(coor[:, 0]*int(math.pow(2, 5-i)) + coor[:, 1], dtype=torch.long)
                    right_target_features_tmp.append(torch.mean(right_image_encoded_features_tmp[:, flatten_coor, :], dim=1))
                right_target_features_tmp = torch.cat(right_target_features_tmp, dim=0)
                right_target_features.append(right_target_features_tmp)

            left_predicted_features = torch.cat(left_predicted_features, dim=1)
            right_predicted_features = torch.cat(right_predicted_features, dim=1)
            left_target_features = torch.cat(left_target_features, dim=1)
            right_target_features = torch.cat(right_target_features, dim=1)


        similarity_map = 1.0 - self.cosine_similarity(left_predicted_features, right_predicted_features).detach().cpu().numpy()
        indices = linear_sum_assignment(similarity_map)

        target = torch.from_numpy(indices[0] == indices[1]).long().to(device)
        features = []
        for idx in range(target.shape[0]):
            if target[idx] == 1:
                features.append(torch.cat([left_target_features[indices[0][idx]], right_target_features[indices[1][idx]]], dim=0).unsqueeze(0))
            else:
                features.append(torch.cat([left_predicted_features[indices[0][idx]], right_predicted_features[indices[1][idx]]], dim=0).unsqueeze(0))
        features = torch.cat(features, dim=0)
        features = self.fc_matching(features)

        loss = F.cross_entropy(features, target)
        return loss

    def safe_division(self, numerator, denominator):
        sign = torch.sign(denominator)
        sign[sign == 0] = 1
        return numerator / (
            sign
            * torch.maximum(
                torch.abs(denominator),
                1e-5 * torch.ones(denominator.shape).type_as(denominator),
            )
        )

    def transform_points(self, transformation_matrix, points, keep_depth=False):
        """
        Transforms points with a transformation matrix.
        """
        shape = points.shape
        if len(shape) == 2:
            transformation_matrix = transformation_matrix.unsqueeze(0)
            points = points.unsqueeze(0)
        points = F.pad(points, (0, 1), value=1).to(device)
        points = torch.einsum("bij,bnj->bni", transformation_matrix, points)
        if keep_depth:
            if len(shape) == 2:
                points = points.squeeze(0)
            return points
        points = self.safe_division(
            points[:, :, :-1],
            repeat(points[:, :, -1], "... -> ... n", n=points.shape[-1] - 1),
        )
        if len(shape) == 2:
            points = points.squeeze(0)
        return points
    
    def transform_points_1_to_2(self, transformation_matrix, points):
        points = points.unsqueeze(0)
        return self.transform_points(transformation_matrix.unsqueeze(0), points, keep_depth=False)[0]
    
    def calculate_l1_and_giou_loss(self, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes):
        # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
        # Left image
        left_indices, left_sizes = self.matcher(left_predicted_bboxes, left_target_bboxes)
        left_num_boxes = sum(left_sizes)
        left_bboxes_loss = self.loss_boxes(left_predicted_bboxes, left_target_bboxes, left_indices, left_num_boxes)
        # Right image
        right_indices, right_sizes = self.matcher(right_predicted_bboxes, right_target_bboxes)
        right_num_boxes = sum(right_sizes)
        right_bboxes_loss = self.loss_boxes(right_predicted_bboxes, right_target_bboxes, right_indices, right_num_boxes)

        return left_bboxes_loss, right_bboxes_loss, left_num_boxes, right_num_boxes

    def loss_matching_boxes(self, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes,
                            left_transformation, right_transformation, left_resize_transformation, right_resize_transformation):
        if not torch.is_tensor(left_predicted_bboxes):
            if isinstance(left_predicted_bboxes, list):
                left_predicted_bboxes = torch.from_numpy(np.asarray(left_predicted_bboxes))
            if isinstance(left_predicted_bboxes, np.ndarray):
                left_predicted_bboxes = torch.from_numpy(left_predicted_bboxes)

        if not torch.is_tensor(right_predicted_bboxes):
            if isinstance(right_predicted_bboxes, list):
                right_predicted_bboxes = torch.from_numpy(np.asarray(right_predicted_bboxes))
            if isinstance(right_predicted_bboxes, np.ndarray):
                right_predicted_bboxes = torch.from_numpy(right_predicted_bboxes)

        # Save a copy of original predicted boxes in left and right images
        left_predicted_bboxes_ori = left_predicted_bboxes
        right_predicted_bboxes_ori = right_predicted_bboxes

        # Step 1: We need to check ground-truth match our requirements. If not, we use normal L1 and GIoU loss
        # If it passed, we move to Step 2.
        # In here, we don't calculate transformation_matrix before making transformation because the aggreation numerical error problem.
        left_target_bboxes_before_resize = self.transform_points_1_to_2(torch.inverse(left_resize_transformation), left_target_bboxes.reshape(-1, 2))
        left_target_bboxes_inv = self.transform_points_1_to_2(torch.inverse(left_transformation), left_target_bboxes_before_resize)
        left_to_right_target_bboxes_before_resize = self.transform_points_1_to_2(right_transformation, left_target_bboxes_inv)
        left_to_right_target_bboxes = self.transform_points_1_to_2(right_resize_transformation, left_to_right_target_bboxes_before_resize).reshape(-1, 4)
        iou_target_bboxes = (box_iou(left_to_right_target_bboxes, right_target_bboxes)[0] > self.args.iou_match_target_threshold).long()
        if torch.sum(torch.diag(iou_target_bboxes, 0)) != min(iou_target_bboxes.shape[:2]):
            return self.calculate_l1_and_giou_loss(left_predicted_bboxes.unsqueeze(0), right_predicted_bboxes.unsqueeze(0), [left_target_bboxes], [right_target_bboxes])

        # Step 2:
        left_predicted_bboxes_before_resize = self.transform_points_1_to_2(torch.inverse(left_resize_transformation), left_predicted_bboxes[:, :4].reshape(-1, 2))
        left_predicted_bboxes_inv = self.transform_points_1_to_2(torch.inverse(left_transformation), left_predicted_bboxes_before_resize)
        left_to_right_predicted_bboxes_before_resize = self.transform_points_1_to_2(right_transformation, left_predicted_bboxes_inv)
        left_to_right_predicted_bboxes = self.transform_points_1_to_2(right_resize_transformation, left_to_right_predicted_bboxes_before_resize).reshape(-1, 4)

        if (left_to_right_predicted_bboxes[:, 2:] >= left_to_right_predicted_bboxes[:, :2]).all():
            iou, union = box_iou(left_to_right_predicted_bboxes, right_predicted_bboxes[:,:4])
            iou_matching = iou > self.args.iou_threshold
            iou_matching = iou_matching.long()
            left_boxes_idxs = torch.where(torch.sum(iou_matching, dim=1) > 0)[0]   # Find all noise boxes in the left image
            right_boxes_idxs = torch.where(torch.sum(iou_matching, dim=0) > 0)[0]   # Find all noise boxes in the right image
            left_predicted_bboxes = left_predicted_bboxes[left_boxes_idxs]
            right_predicted_bboxes = right_predicted_bboxes[right_boxes_idxs]

            # If we cannot find any matching pair between boxes in left and right image
            # It means that our model is not enough good and we need to force the model to improve object detection
            # We will use the normal L1 and GIoU loss
            if len(left_predicted_bboxes) == 0 or len(right_predicted_bboxes) == 0:
                return self.calculate_l1_and_giou_loss(left_predicted_bboxes_ori.unsqueeze(0), right_predicted_bboxes_ori.unsqueeze(0), [left_target_bboxes], [right_target_bboxes])

            left_remove_index = []
            right_remove_index = []
            left_remove_target_index = []
            right_remove_target_index = []

            left_predicted_bboxes_before_resize = self.transform_points_1_to_2(torch.inverse(left_resize_transformation), left_predicted_bboxes[:, :4].reshape(-1, 2))
            left_predicted_bboxes_inv = self.transform_points_1_to_2(torch.inverse(left_transformation), left_predicted_bboxes_before_resize)
            left_to_right_predicted_bboxes_before_resize = self.transform_points_1_to_2(right_transformation, left_predicted_bboxes_inv)
            left_to_right_predicted_bboxes = self.transform_points_1_to_2(right_resize_transformation, left_to_right_predicted_bboxes_before_resize).reshape(-1, 4)
            iou, union = box_iou(left_to_right_predicted_bboxes, right_predicted_bboxes[:,:4])
            iou_matching_mask = iou > self.args.iou_match_threshold
            iou_matching_mask = iou_matching_mask.long()

            # Find maximum matching in each row
            iou_max_row = torch.argmax(iou_matching_mask, dim=1)
            iou_max_col = torch.argmax(iou_matching_mask, dim=0)
            for i, idx in enumerate(iou_max_row):
                if iou_matching_mask[i, idx] == False:
                    continue
                else:
                    if iou_max_col[idx] == i:
                        left_box_target_iou = box_iou(left_predicted_bboxes[i].unsqueeze(0), left_target_bboxes)[0] 
                        left_box_target_iou_mask = left_box_target_iou > self.args.iou_match_target_threshold
                        left_box_target_index = torch.argmax(left_box_target_iou)
                        right_box_target_iou = box_iou(right_predicted_bboxes[idx].unsqueeze(0), right_target_bboxes)[0] 
                        right_box_target_iou_mask = right_box_target_iou > self.args.iou_match_target_threshold
                        right_box_target_index = torch.argmax(right_box_target_iou)

                        if (left_box_target_index == right_box_target_index).all() and left_box_target_iou_mask[0][left_box_target_index] == True and right_box_target_iou_mask[0][right_box_target_index] == True:
                            left_remove_index.append(i)
                            right_remove_index.append(idx)
                            left_remove_target_index.append(left_box_target_index)
                            right_remove_target_index.append(right_box_target_index)

            left_keep_index = [i for i in range(len(left_predicted_bboxes)) if i not in left_remove_index]
            right_keep_index = [i for i in range(len(right_predicted_bboxes)) if i not in right_remove_index]
            left_keep_target_index = [i for i in range(len(left_target_bboxes)) if i not in left_remove_target_index]
            right_keep_target_index = [i for i in range(len(right_target_bboxes)) if i not in right_remove_target_index]

            # Now we have set. Set1: Set of selected boxes. Set2: Set of remove boxes
            # We need to define Loss for each Set1 and Set2
            # We made punishment for all boxes cannot find the matching pairs of boxes
            left_predicted_bboxes_keep = left_predicted_bboxes[left_keep_index]
            right_predicted_bboxes_keep = right_predicted_bboxes[right_keep_index] 
            left_target_bboxes_keep = left_target_bboxes[left_keep_target_index]
            right_target_bboxes_keep = right_target_bboxes[right_keep_target_index]
            if len(left_target_bboxes_keep) == 0 or len(right_target_bboxes_keep) == 0:
                left_losses = {}
                left_losses['loss_bbox'] = 0.0
                left_losses['loss_giou'] = 0.0
                right_losses = {}
                right_losses['loss_bbox'] = 0.0
                right_losses['loss_giou'] = 0.0

                return left_losses, right_losses, 0, 0

            # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
            return self.calculate_l1_and_giou_loss(left_predicted_bboxes_keep.unsqueeze(0), right_predicted_bboxes_keep.unsqueeze(0), [left_target_bboxes_keep], [right_target_bboxes_keep])
        else:
            # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
            return self.calculate_l1_and_giou_loss(left_predicted_bboxes.unsqueeze(0), right_predicted_bboxes.unsqueeze(0), [left_target_bboxes], [right_target_bboxes])

    def training_step(self, batch, batch_idx):
        left_image_outputs, right_image_outputs, left_image_encoded_features, right_image_encoded_features = self(batch)
        # Loss from centernet
        left_losses = self.centernet_head.loss(
            *left_image_outputs,
            batch["left_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        right_losses = self.centernet_head.loss(
            *right_image_outputs,
            batch["right_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )

        # Select top 100 predict boxes with highest confidence score for left and right images
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )

        if self.args.loss_type == 'detr_and_centernet':
            left_predicted_bboxes_refine = []
            for i, bboxes in enumerate(left_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    left_predicted_bboxes_refine.append(new_bboxes)
                else:
                    left_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            right_predicted_bboxes_refine = []
            for i, bboxes in enumerate(right_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    right_predicted_bboxes_refine.append(new_bboxes)
                else:
                    right_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            left_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in left_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)
            right_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in right_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)

            # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
            # Left image
            left_indices, left_sizes = self.matcher(left_predicted_bboxes, batch["left_image_target_bboxes"])
            left_num_boxes = sum(left_sizes)
            left_bboxes_loss = self.loss_boxes(left_predicted_bboxes, batch["left_image_target_bboxes"], left_indices, left_num_boxes)
            # Right image
            right_indices, right_sizes = self.matcher(right_predicted_bboxes, batch["right_image_target_bboxes"])
            right_num_boxes = sum(right_sizes)
            right_bboxes_loss = self.loss_boxes(right_predicted_bboxes, batch["right_image_target_bboxes"], right_indices, right_num_boxes)

            overall_loss = 0
            for key in left_losses:
                self.log(
                    f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
                )
                self.log("train/left_bboxes_loss", left_bboxes_loss['loss_bbox'] + left_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
                self.log("train/right_bboxes_loss", right_bboxes_loss['loss_bbox'] + right_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
                overall_loss += left_losses[key] + right_losses[key] + \
                    left_bboxes_loss['loss_bbox'] + left_bboxes_loss['loss_giou'] + \
                    right_bboxes_loss['loss_bbox'] + right_bboxes_loss['loss_giou']
            self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
            return overall_loss
        elif self.args.loss_type == 'l1_and_giou':
            left_predicted_bboxes_refine = []
            for i, bboxes in enumerate(left_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    left_predicted_bboxes_refine.append(new_bboxes)
                else:
                    left_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            right_predicted_bboxes_refine = []
            for i, bboxes in enumerate(right_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    right_predicted_bboxes_refine.append(new_bboxes)
                else:
                    right_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            left_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in left_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)
            right_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in right_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)

            # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
            # Left image
            left_indices, left_sizes = self.matcher(left_predicted_bboxes, batch["left_image_target_bboxes"])
            left_num_boxes = sum(left_sizes)
            left_bboxes_loss = self.loss_boxes(left_predicted_bboxes, batch["left_image_target_bboxes"], left_indices, left_num_boxes)
            # Right image
            right_indices, right_sizes = self.matcher(right_predicted_bboxes, batch["right_image_target_bboxes"])
            right_num_boxes = sum(right_sizes)
            right_bboxes_loss = self.loss_boxes(right_predicted_bboxes, batch["right_image_target_bboxes"], right_indices, right_num_boxes)

            overall_loss = 0
            for key in left_losses:
                self.log(
                    f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
                )
            self.log("train/left_bboxes_loss", left_bboxes_loss['loss_bbox'] + left_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
            self.log("train/right_bboxes_loss", right_bboxes_loss['loss_bbox'] + right_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
            overall_loss += self.args.lamda_l1*left_bboxes_loss['loss_bbox'] + self.args.lamda_giou*left_bboxes_loss['loss_giou'] + \
                self.args.lamda_l1*right_bboxes_loss['loss_bbox'] + self.args.lamda_giou*right_bboxes_loss['loss_giou']
            self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
            return overall_loss
        
        elif self.args.loss_type == 'centernet_l1_giou_matching':
            left_predicted_bboxes_refine = []
            for i, bboxes in enumerate(left_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    left_predicted_bboxes_refine.append(new_bboxes)
                else:
                    left_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            right_predicted_bboxes_refine = []
            for i, bboxes in enumerate(right_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    right_predicted_bboxes_refine.append(new_bboxes)
                else:
                    right_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            left_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in left_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)
            right_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in right_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)

            # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
            # Left image
            left_indices, left_sizes = self.matcher(left_predicted_bboxes, batch["left_image_target_bboxes"])
            left_num_boxes = sum(left_sizes)
            left_bboxes_loss = self.loss_boxes(left_predicted_bboxes, batch["left_image_target_bboxes"], left_indices, left_num_boxes)
            # Right image
            right_indices, right_sizes = self.matcher(right_predicted_bboxes, batch["right_image_target_bboxes"])
            right_num_boxes = sum(right_sizes)
            right_bboxes_loss = self.loss_boxes(right_predicted_bboxes, batch["right_image_target_bboxes"], right_indices, right_num_boxes)

            # Calculate loss for matching pair between left and right images
            left_images = batch['left_image'].to(device)
            right_images = batch['right_image'].to(device)
            loss_matches = 0
            for idx, images in enumerate(zip(left_images, right_images)):
                loss_matches += self.loss_match(images[0], images[1], left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx])
            loss_matches = loss_matches/(left_predicted_bboxes.shape[0] + 1e-8)

            overall_loss = 0
            for key in left_losses:
                self.log(
                    f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
                )
                overall_loss += left_losses[key] + right_losses[key]
            overall_loss += self.args.lamda_l1*left_bboxes_loss['loss_bbox'] + self.args.lamda_giou*left_bboxes_loss['loss_giou'] + \
                self.args.lamda_l1*right_bboxes_loss['loss_bbox'] + self.args.lamda_giou*right_bboxes_loss['loss_giou']
            overall_loss += loss_matches

            self.log("train/left_bboxes_loss", left_bboxes_loss['loss_bbox'] + left_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
            self.log("train/right_bboxes_loss", right_bboxes_loss['loss_bbox'] + right_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
            self.log("train/matching_loss", loss_matches, on_step=True, on_epoch=True)
            self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
            return overall_loss
        
        elif self.args.loss_type == 'centernet_matching':
            left_predicted_bboxes_refine = []
            for i, bboxes in enumerate(left_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    left_predicted_bboxes_refine.append(new_bboxes)
                else:
                    left_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            right_predicted_bboxes_refine = []
            for i, bboxes in enumerate(right_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    right_predicted_bboxes_refine.append(new_bboxes)
                else:
                    right_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            left_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in left_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)
            right_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in right_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)

            # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
            # Left image
            left_indices, left_sizes = self.matcher(left_predicted_bboxes, batch["left_image_target_bboxes"])
            left_num_boxes = sum(left_sizes)
            left_bboxes_loss = self.loss_boxes(left_predicted_bboxes, batch["left_image_target_bboxes"], left_indices, left_num_boxes)
            # Right image
            right_indices, right_sizes = self.matcher(right_predicted_bboxes, batch["right_image_target_bboxes"])
            right_num_boxes = sum(right_sizes)
            right_bboxes_loss = self.loss_boxes(right_predicted_bboxes, batch["right_image_target_bboxes"], right_indices, right_num_boxes)

            # Calculate loss for matching pair between left and right images
            loss_matches = 0
            if self.args.use_ground_truth == 'pure_ground_truth':
                left_target_bboxes = batch["left_image_target_bboxes"]
                right_target_bboxes = batch["right_image_target_bboxes"]
                for idx in range(len(left_target_bboxes)):
                    if not self.args.use_coattention_feature:
                        loss_matches += self.loss_match_with_context_ground_truth(left_target_bboxes[idx], right_target_bboxes[idx], [left_image_encoded_features[idx]], [right_image_encoded_features[idx]])
                    else:
                        loss_matches += self.loss_match_with_context_ground_truth(left_target_bboxes[idx], right_target_bboxes[idx], 
                                                                                  [left_image_encoded_features[0][idx], left_image_encoded_features[1][idx], left_image_encoded_features[2][idx]], 
                                                                                  [right_image_encoded_features[0][idx], right_image_encoded_features[1][idx], right_image_encoded_features[2][idx]])
                loss_matches = loss_matches/(len(left_target_bboxes) + 1e-8)
            elif self.args.use_ground_truth == 'mix_ground_truth':
                left_target_bboxes = batch["left_image_target_bboxes"]
                right_target_bboxes = batch["right_image_target_bboxes"]
                for idx in range(left_predicted_bboxes.shape[0]):
                    if not self.args.use_coattention_feature:
                        loss_matches += self.loss_match_with_context_mix_ground_truth(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx],
                                                                                    left_target_bboxes[idx], right_target_bboxes[idx], 
                                                                                    left_image_encoded_features[idx], 
                                                                                    right_image_encoded_features[idx])
                    else:
                        loss_matches += self.loss_match_with_context_mix_ground_truth(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx],
                                                                                    left_target_bboxes[idx], right_target_bboxes[idx], 
                                                                                    [left_image_encoded_features[0][idx], left_image_encoded_features[1][idx], left_image_encoded_features[2][idx]], 
                                                                                    [right_image_encoded_features[0][idx], right_image_encoded_features[1][idx], right_image_encoded_features[2][idx]])
                loss_matches = loss_matches/(left_predicted_bboxes.shape[0] + 1e-8)
            else:
                for idx in range(left_predicted_bboxes.shape[0]):
                    if not self.args.use_coattention_feature:
                        loss_matches += self.loss_match_with_context(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx], left_image_encoded_features[idx], right_image_encoded_features[idx])
                    else:
                        loss_matches += self.loss_match_with_context(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx], 
                                                                     [left_image_encoded_features[0][idx], left_image_encoded_features[1][idx], left_image_encoded_features[2][idx]], 
                                                                     [right_image_encoded_features[0][idx], right_image_encoded_features[1][idx], right_image_encoded_features[2][idx]])
                loss_matches = loss_matches/(left_predicted_bboxes.shape[0] + 1e-8)

            overall_loss = 0
            for key in left_losses:
                self.log(
                    f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
                )
                overall_loss += left_losses[key] + right_losses[key]

            overall_loss += self.args.lamda_matching*loss_matches

            self.log("train/left_bboxes_loss", left_bboxes_loss['loss_bbox'] + left_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
            self.log("train/right_bboxes_loss", right_bboxes_loss['loss_bbox'] + right_bboxes_loss['loss_giou'], on_step=True, on_epoch=True)
            self.log("train/matching_loss", loss_matches, on_step=True, on_epoch=True)
            self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
            return overall_loss
        
        elif self.args.loss_type == 'post_processing':
            left_predicted_bboxes_refine = []
            for i, bboxes in enumerate(left_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    left_predicted_bboxes_refine.append(new_bboxes)
                else:
                    left_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            right_predicted_bboxes_refine = []
            for i, bboxes in enumerate(right_predicted_bboxes):
                new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
                if len(new_bboxes) < top_n_bboxes:
                    new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
                if len(new_bboxes) > 0:
                    new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                    right_predicted_bboxes_refine.append(new_bboxes)
                else:
                    right_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

            left_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in left_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)
            right_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in right_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)
            left_target_bboxes = batch["left_image_target_bboxes"]
            right_target_bboxes = batch["right_image_target_bboxes"]
            left_transform_matrix = batch["left_transformation"]
            right_transform_matrix = batch["right_transformation"]
            left_resize_transformation = batch["left_resize_transformation"]
            right_resize_transformation = batch["right_resize_transformation"]

            left_bboxes_loss_avg = []
            right_bboxes_loss_avg = []
            left_num_boxes_avg = []
            right_num_boxes_avg = []
            for idx in range(left_predicted_bboxes.shape[0]):
                left_bboxes_loss, right_bboxes_loss, left_num_boxes, right_num_boxes = self.loss_matching_boxes(left_predicted_bboxes[idx], right_predicted_bboxes[idx], 
                                                                                                                left_target_bboxes[idx], right_target_bboxes[idx], 
                                                                                                                left_transform_matrix[idx], right_transform_matrix[idx],
                                                                                                                left_resize_transformation[idx], right_resize_transformation[idx])
                left_bboxes_loss_avg.append(left_bboxes_loss['loss_bbox']*left_num_boxes + left_bboxes_loss['loss_giou']*left_num_boxes)
                left_num_boxes_avg.append(left_num_boxes)
                right_bboxes_loss_avg.append(right_bboxes_loss['loss_bbox']*right_num_boxes + right_bboxes_loss['loss_giou']*right_num_boxes)
                right_num_boxes_avg.append(right_num_boxes)
            left_bboxes_loss_avg = sum(left_bboxes_loss_avg)/(sum(left_num_boxes_avg) + 1e-8)
            right_bboxes_loss_avg = sum(right_bboxes_loss_avg)/(sum(right_num_boxes_avg) + 1e-8)

            overall_loss = 0
            for key in left_losses:
                self.log(
                    f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
                )
                self.log("train/left_bboxes_loss", left_bboxes_loss_avg, on_step=True, on_epoch=True)
                self.log("train/right_bboxes_loss", right_bboxes_loss_avg, on_step=True, on_epoch=True)
                overall_loss += left_losses[key] + right_losses[key] + \
                    left_bboxes_loss_avg + \
                    right_bboxes_loss_avg
            self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
            return overall_loss
        else:
            overall_loss = 0
            for key in left_losses:
                self.log(
                    f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
                )
                overall_loss += left_losses[key] + right_losses[key]
            self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
            return overall_loss

    def validation_step(self, batch, batch_index):
        left_image_outputs, right_image_outputs, left_image_encoded_features, right_image_encoded_features = self(batch)
        left_losses = self.centernet_head.loss(
            *left_image_outputs,
            batch["left_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        right_losses = self.centernet_head.loss(
            *right_image_outputs,
            batch["right_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )

        # Select top 100 predict boxes with highest confidence score for left and right images
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )

        left_predicted_bboxes_refine = []
        for i, bboxes in enumerate(left_predicted_bboxes):
            new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
            if len(new_bboxes) < top_n_bboxes:
                new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
            if len(new_bboxes) > 0:
                new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                left_predicted_bboxes_refine.append(new_bboxes)
            else:
                left_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

        right_predicted_bboxes_refine = []
        for i, bboxes in enumerate(right_predicted_bboxes):
            new_bboxes = [bbox.unsqueeze(0) for bbox in bboxes[0] if bbox[2]-bbox[0] > 0 and bbox[3]-bbox[1]>0]
            if len(new_bboxes) < top_n_bboxes:
                new_bboxes.extend([torch.zeros((1,5)).to(device)]*(top_n_bboxes-len(new_bboxes)))
            if len(new_bboxes) > 0:
                new_bboxes = torch.cat(new_bboxes)[:top_n_bboxes]
                right_predicted_bboxes_refine.append(new_bboxes)
            else:
                right_predicted_bboxes_refine.append(torch.zeros((top_n_bboxes, 5)).to(device))

        left_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in left_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)
        right_predicted_bboxes = torch.cat([bboxes[:,:4] for bboxes in right_predicted_bboxes_refine]).reshape(-1, top_n_bboxes, 4)

        overall_loss = 0
        for key in left_losses:
            self.log(f"val/{key}", left_losses[key] + right_losses[key], on_epoch=True)
            overall_loss += left_losses[key] + right_losses[key]

        # Using Hungarian algorithm to find the matching between predicted boxes and target boxes
        # Left image
        left_indices, left_sizes = self.matcher(left_predicted_bboxes, batch["left_image_target_bboxes"])
        # Right image
        right_indices, right_sizes = self.matcher(right_predicted_bboxes, batch["right_image_target_bboxes"])

        # Calculate loss for matching pair between left and right images
        loss_matches = 0.0
        if self.args.use_ground_truth == 'pure_ground_truth':
            left_target_bboxes = batch["left_image_target_bboxes"]
            right_target_bboxes = batch["right_image_target_bboxes"]
            for idx in range(len(left_target_bboxes)):
                if not self.args.use_coattention_feature:
                    loss_matches += self.loss_match_with_context_ground_truth(left_target_bboxes[idx], right_target_bboxes[idx], [left_image_encoded_features[idx]], [right_image_encoded_features[idx]])
                else:
                    loss_matches += self.loss_match_with_context_ground_truth(left_target_bboxes[idx], right_target_bboxes[idx], 
                                                                            [left_image_encoded_features[0][idx], left_image_encoded_features[1][idx], left_image_encoded_features[2][idx]], 
                                                                            [right_image_encoded_features[0][idx], right_image_encoded_features[1][idx], right_image_encoded_features[2][idx]])
            loss_matches = loss_matches/(len(left_target_bboxes) + 1e-8)
            self.log("val/matching_loss", loss_matches, on_step=True, on_epoch=True)
        elif self.args.use_ground_truth == 'mix_ground_truth':
            left_target_bboxes = batch["left_image_target_bboxes"]
            right_target_bboxes = batch["right_image_target_bboxes"]
            for idx in range(left_predicted_bboxes.shape[0]):
                if not self.args.use_coattention_feature:
                    loss_matches += self.loss_match_with_context_mix_ground_truth(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx],
                                                                                  left_target_bboxes[idx], right_target_bboxes[idx], [left_image_encoded_features[idx]], [right_image_encoded_features[idx]])
                else:
                    loss_matches += self.loss_match_with_context_mix_ground_truth(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx],
                                                                                left_target_bboxes[idx], right_target_bboxes[idx], 
                                                                                [left_image_encoded_features[0][idx], left_image_encoded_features[1][idx], left_image_encoded_features[2][idx]], 
                                                                                [right_image_encoded_features[0][idx], right_image_encoded_features[1][idx], right_image_encoded_features[2][idx]])
            loss_matches = loss_matches/(left_predicted_bboxes.shape[0] + 1e-8)
            self.log("val/matching_loss", loss_matches, on_step=True, on_epoch=True)
        elif self.args.use_ground_truth == 'pure_predicted_boxes':
            for idx in range(left_predicted_bboxes.shape[0]):
                if not self.args.use_coattention_feature:
                    loss_matches += self.loss_match_with_context(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx], [left_image_encoded_features[idx]], [right_image_encoded_features[idx]])
                else:
                    loss_matches += self.loss_match_with_context(left_predicted_bboxes[idx], right_predicted_bboxes[idx], left_indices[idx], right_indices[idx], 
                                                                [left_image_encoded_features[0][idx], left_image_encoded_features[1][idx], left_image_encoded_features[2][idx]], 
                                                                [right_image_encoded_features[0][idx], right_image_encoded_features[1][idx], right_image_encoded_features[2][idx]])
            loss_matches = loss_matches/(left_predicted_bboxes.shape[0] + 1e-8)
            self.log("val/matching_loss", loss_matches, on_step=True, on_epoch=True)
        else:
            loss_matches = 0.0

        overall_loss += self.args.lamda_matching*loss_matches
        self.log("val/overall_loss", overall_loss, on_epoch=True)

        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )

        return left_predicted_bboxes, right_predicted_bboxes
    
    def test_step(self, batch, batch_index, dataloader_index=0):
        left_image_outputs, right_image_outputs, left_image_encoded_features_last, right_image_encoded_features_last = self(batch)
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )

        # left_predicted_bboxes = get_bboxes(self.test_cfg,
        #     *left_image_outputs,
        #     img_metas=batch["query_metadata"],
        #     rescale=False,
        # )
        # right_predicted_bboxes = get_bboxes(self.test_cfg,
        #     *right_image_outputs,
        #     img_metas=batch["query_metadata"],
        #     rescale=False,
        # )

        """
        # Using NMS and select topk most confidence score boxes
        if self.args.wacv:
            left_predicted_bboxes_nms = []
            for bboxes, classifcation in left_predicted_bboxes:
                bboxes = bboxes.cpu().numpy()
                bboxes, maximum = suppress_non_maximum(bboxes)
                bboxes = np.asarray(bboxes)
                bboxes = bboxes[bboxes[:, 4].argsort()][::-1][:self.args.topk, :]
                bboxes = torch.from_numpy(bboxes.copy())
                maximum = torch.from_numpy(maximum) == 1
                classifcation = classifcation.cpu()
                classifcation = classifcation[maximum][bboxes[:, 4].argsort()][:self.args.topk]
                left_predicted_bboxes_nms.append([bboxes, classifcation])

            right_predicted_bboxes_nms = []
            for bboxes, classifcation in right_predicted_bboxes:
                bboxes = bboxes.cpu().numpy()
                bboxes, maximum = suppress_non_maximum(bboxes)
                bboxes = np.asarray(bboxes)
                bboxes = bboxes[bboxes[:, 4].argsort()][::-1][:self.args.topk, :]
                bboxes = torch.from_numpy(bboxes.copy())
                maximum = torch.from_numpy(maximum) == 1
                classifcation = classifcation.cpu()
                classifcation = classifcation[maximum][bboxes[:, 4].argsort()][:self.args.topk]
                right_predicted_bboxes_nms.append([bboxes, classifcation])

            return (left_predicted_bboxes_nms,
                    right_predicted_bboxes_nms, 
                    [bboxes.cpu() for bboxes in batch["left_image_target_bboxes"]],
                    [bboxes.cpu() for bboxes in batch["right_image_target_bboxes"]])
        """

        return (
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in left_predicted_bboxes
            ],
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in right_predicted_bboxes
            ],
            [bboxes.cpu() for bboxes in batch["left_image_target_bboxes"]],
            [bboxes.cpu() for bboxes in batch["right_image_target_bboxes"]],
            [left_image_encoded_features_last],
            [right_image_encoded_features_last]
        )

        # return (
        #     [
        #         [bboxes[torch.where(bboxes[:,4] > 0.0)].cpu(), classification[torch.where(bboxes[:,4] > 0.0)].cpu()]
        #         for bboxes, classification in left_predicted_bboxes
        #     ],
        #     [
        #         [bboxes[torch.where(bboxes[:,4] > 0.0)].cpu(), classification[torch.where(bboxes[:,4] > 0.0)].cpu()]
        #         for bboxes, classification in right_predicted_bboxes
        #     ],
        #     [bboxes.cpu() for bboxes in batch["left_image_target_bboxes"]],
        #     [bboxes.cpu() for bboxes in batch["right_image_target_bboxes"]],
        # )

    def test_step_emd(self, emd_model, batch, batch_index, dataloader_index=0):
        left_image_outputs, right_image_outputs, _, _ = self(batch)
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )

        left_predicted_bboxes_refine = []
        right_predicted_bboxes_refine = []
        left_image_target_bboxes = []
        right_image_target_bboxes = []
        for index in range(len(left_predicted_bboxes)):
            left_image_path = batch["left_image_path"][index]
            right_image_path = batch["right_image_path"][index]
            if left_image_path is not None:
                left_image = Image.open(left_image_path).convert('RGB')
            else:
                left_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["left_image"][0]))).convert('RGB')
            if right_image_path is not None:
                right_image = Image.open(right_image_path).convert('RGB')
            else:
                right_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["right_image"][0]))).convert('RGB')

            # Select only top-k bounding boxes with the most confidence
            left_bboxes = left_predicted_bboxes[index][0].detach().cpu().numpy()
            right_bboxes = right_predicted_bboxes[index][0].detach().cpu().numpy()
            left_classes = left_predicted_bboxes[index][1].detach().cpu().numpy()
            right_classes = right_predicted_bboxes[index][1].detach().cpu().numpy()
            left_classes = left_classes[left_bboxes[:, 4].argsort()][::-1][:self.args.topk]
            right_classes = right_classes[right_bboxes[:, 4].argsort()][::-1][:self.args.topk]
            left_bboxes_sort = left_bboxes[left_bboxes[:, 4].argsort()][::-1][:self.args.topk]
            right_bboxes_sort = right_bboxes[right_bboxes[:, 4].argsort()][::-1][:self.args.topk]

            if self.args.use_nms:
                _, left_maximum = suppress_non_maximum(
                    left_bboxes_sort[:self.args.topk, :4]
                )
                _, right_maximum = suppress_non_maximum(
                    right_bboxes_sort[:self.args.topk, :4]
                )
                left_bboxes_nms = []
                right_bboxes_nms = []
                left_classes_nms = []
                right_classes_nms = []
                for idx in range(len(left_maximum)):
                    if left_maximum[idx] == 1:
                        left_bboxes_nms.append(left_bboxes_sort[idx])
                        left_classes_nms.append(left_classes[idx])

                for idx in range(len(right_maximum)):
                    if right_maximum[idx] == 1:
                        right_bboxes_nms.append(right_bboxes_sort[idx])
                        right_classes_nms.append(right_classes[idx])

                # Remove trivial (1 pixel) boxes
                left_bboxes_nms_refine = [bbox for bbox in left_bboxes_nms
                                         if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                right_bboxes_nms_refine = [bbox for bbox in right_bboxes_nms
                                         if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
            else:
                left_bboxes_nms_refine = [bbox for bbox in left_bboxes_sort
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                right_bboxes_nms_refine = [bbox for bbox in right_bboxes_sort
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
            
            # Remove wrong bounding boxes
            left_bboxes_nms_refine = [bbox for bbox in left_bboxes_nms_refine
                                     if bbox[2]-bbox[0] > 0 and bbox[3] - bbox[1] > 0]
            right_bboxes_nms_refine = [bbox for bbox in right_bboxes_nms_refine
                                     if bbox[2]-bbox[0] > 0 and bbox[3] - bbox[1] > 0]
            
            # Select boxes above confidence threshold
            left_bboxes_nms_refine = [bbox for bbox in left_bboxes_nms_refine
                                     if bbox[4] >= self.args.det_thres]
            right_bboxes_nms_refine = [bbox for bbox in right_bboxes_nms_refine
                                     if bbox[4] >= self.args.det_thres]


            # Move to next if non-exist bounding box in one of two images
            if len(left_bboxes_nms_refine) == 0 or len(right_bboxes_nms_refine) == 0:
                if len(left_bboxes_nms_refine) == 0:
                    left_predicted_emd_hungarian = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.01]])
                else:
                    left_predicted_emd_hungarian = torch.cat([torch.from_numpy(bbox).unsqueeze(0) for bbox in left_bboxes_nms_refine], dim=0)
                if len(right_bboxes_nms_refine) == 0:
                    right_predicted_emd_hungarian = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.01]])
                else:
                    right_predicted_emd_hungarian = torch.cat([torch.from_numpy(bbox).unsqueeze(0) for bbox in right_bboxes_nms_refine], dim=0)
                left_predicted_classes = torch.zeros(len(left_predicted_emd_hungarian))
                right_predicted_classes = torch.zeros(len(right_predicted_emd_hungarian))
                left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
                right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
                left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
                right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())
                continue

            left_crop_bboxes = torch.cat([emd_transform(left_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in left_bboxes_nms_refine], dim=0)
            right_crop_bboxes = torch.cat([emd_transform(right_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in right_bboxes_nms_refine], dim=0)
            data = torch.cat((left_crop_bboxes, right_crop_bboxes)).to(device)

            emd_model.mode = 'encoder'
            data = emd_model(data)
            k = left_crop_bboxes.shape[0]
            data_shot, data_query = data[:k], data[k:]
            similarity_map = 1.0 - emd_model.get_similiarity_map(data_shot, data_query)
            similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
            indices = linear_sum_assignment(similarity_map)
            indices = [indices[1], indices[0]]

            left_predicted_emd_hungarian = torch.cat([torch.from_numpy(left_bboxes_nms_refine[idx]).unsqueeze(0) for idx in indices[0]], dim=0)
            right_predicted_emd_hungarian = torch.cat([torch.from_numpy(right_bboxes_nms_refine[idx]).unsqueeze(0) for idx in indices[1]], dim=0)
            if self.args.use_nms:
                left_predicted_classes = torch.from_numpy(np.asarray([left_classes_nms[idx] for idx in indices[0]]))
                right_predicted_classes = torch.from_numpy(np.asarray([right_classes_nms[idx] for idx in indices[1]]))
            else:
                left_predicted_classes = torch.from_numpy(np.asarray([left_classes[idx] for idx in indices[0]]))
                right_predicted_classes = torch.from_numpy(np.asarray([right_classes[idx] for idx in indices[1]]))
            left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
            right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
            left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
            right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())

        return (
            left_predicted_bboxes_refine,
            right_predicted_bboxes_refine,
            left_image_target_bboxes,
            right_image_target_bboxes,
        )

    def test_step_unet_encoder(self, emd_model, batch, batch_index, dataloader_index=0):
        left_image_outputs, right_image_outputs, left_image_encoded_features_last, right_image_encoded_features_last = self(batch)
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )

        left_predicted_bboxes_refine = []
        right_predicted_bboxes_refine = []
        left_image_target_bboxes = []
        right_image_target_bboxes = []
        for index in range(len(left_predicted_bboxes)):
            left_image_path = batch["left_image_path"][index]
            right_image_path = batch["right_image_path"][index]
            if left_image_path is not None:
                left_image = Image.open(left_image_path).convert('RGB')
            else:
                left_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["left_image"][0]))).convert('RGB')
            if right_image_path is not None:
                right_image = Image.open(right_image_path).convert('RGB')
            else:
                right_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["right_image"][0]))).convert('RGB')

            # Select only top-k bounding boxes with the most confidence
            left_bboxes = left_predicted_bboxes[index][0].detach().cpu().numpy()
            right_bboxes = right_predicted_bboxes[index][0].detach().cpu().numpy()
            left_classes = left_predicted_bboxes[index][1].detach().cpu().numpy()
            right_classes = right_predicted_bboxes[index][1].detach().cpu().numpy()
            left_classes = left_classes[left_bboxes[:, 4].argsort()][::-1][:self.args.topk]
            right_classes = right_classes[right_bboxes[:, 4].argsort()][::-1][:self.args.topk]
            left_bboxes_sort = left_bboxes[left_bboxes[:, 4].argsort()][::-1][:self.args.topk]
            right_bboxes_sort = right_bboxes[right_bboxes[:, 4].argsort()][::-1][:self.args.topk]

            if self.args.use_nms:
                _, left_maximum = suppress_non_maximum(
                    left_bboxes_sort[:self.args.topk, :4]
                )
                _, right_maximum = suppress_non_maximum(
                    right_bboxes_sort[:self.args.topk, :4]
                )
                left_bboxes_nms = []
                right_bboxes_nms = []
                left_classes_nms = []
                right_classes_nms = []
                for idx in range(len(left_maximum)):
                    if left_maximum[idx] == 1:
                        left_bboxes_nms.append(left_bboxes_sort[idx])
                        left_classes_nms.append(left_classes[idx])

                for idx in range(len(right_maximum)):
                    if right_maximum[idx] == 1:
                        right_bboxes_nms.append(right_bboxes_sort[idx])
                        right_classes_nms.append(right_classes[idx])

                # Remove trivial (1 pixel) boxes
                left_bboxes_nms_refine = [bbox for bbox in left_bboxes_nms
                                         if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                right_bboxes_nms_refine = [bbox for bbox in right_bboxes_nms
                                         if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
            else:
                left_bboxes_nms_refine = [bbox for bbox in left_bboxes_sort
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
                right_bboxes_nms_refine = [bbox for bbox in right_bboxes_sort
                                        if np.abs(bbox[0]-bbox[2]) > 1 or np.abs(bbox[1] - bbox[3]) > 1]
            
            # Remove wrong bounding boxes
            left_bboxes_nms_refine = [bbox for bbox in left_bboxes_nms_refine
                                     if bbox[2]-bbox[0] > 0 and bbox[3] - bbox[1] > 0]
            right_bboxes_nms_refine = [bbox for bbox in right_bboxes_nms_refine
                                     if bbox[2]-bbox[0] > 0 and bbox[3] - bbox[1] > 0]
            
            # Select boxes above confidence threshold
            left_bboxes_nms_refine = [bbox for bbox in left_bboxes_nms_refine
                                     if bbox[4] >= self.args.det_thres]
            right_bboxes_nms_refine = [bbox for bbox in right_bboxes_nms_refine
                                     if bbox[4] >= self.args.det_thres]


            # Move to next if non-exist bounding box in one of two images
            if len(left_bboxes_nms_refine) == 0 or len(right_bboxes_nms_refine) == 0:
                if len(left_bboxes_nms_refine) == 0:
                    left_predicted_emd_hungarian = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.01]])
                else:
                    left_predicted_emd_hungarian = torch.cat([torch.from_numpy(bbox).unsqueeze(0) for bbox in left_bboxes_nms_refine], dim=0)
                if len(right_bboxes_nms_refine) == 0:
                    right_predicted_emd_hungarian = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.01]])
                else:
                    right_predicted_emd_hungarian = torch.cat([torch.from_numpy(bbox).unsqueeze(0) for bbox in right_bboxes_nms_refine], dim=0)
                left_predicted_classes = torch.zeros(len(left_predicted_emd_hungarian))
                right_predicted_classes = torch.zeros(len(right_predicted_emd_hungarian))
                left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
                right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
                left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
                right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())
                continue

            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
            left_predicted_features = torch.cat([torch.flatten(pooling(self.unet_model.encoder(cyws_transform(left_image.crop(bbox[:4])).unsqueeze(0).to(device))[-1]), 1) for bbox in left_bboxes_nms_refine], dim=0)
            right_predicted_features = torch.cat([torch.flatten(pooling(self.unet_model.encoder(cyws_transform(right_image.crop(bbox[:4])).unsqueeze(0).to(device))[-1]), 1) for bbox in right_bboxes_nms_refine], dim=0)
            similarity_map = 1.0 - self.cosine_similarity(left_predicted_features, right_predicted_features).detach().cpu().numpy()
            indices = linear_sum_assignment(similarity_map)

            left_predicted_emd_hungarian = torch.cat([torch.from_numpy(left_bboxes_nms_refine[idx]).unsqueeze(0) for idx in indices[0]], dim=0)
            right_predicted_emd_hungarian = torch.cat([torch.from_numpy(right_bboxes_nms_refine[idx]).unsqueeze(0) for idx in indices[1]], dim=0)
            if self.args.use_nms:
                left_predicted_classes = torch.from_numpy(np.asarray([left_classes_nms[idx] for idx in indices[0]]))
                right_predicted_classes = torch.from_numpy(np.asarray([right_classes_nms[idx] for idx in indices[1]]))
            else:
                left_predicted_classes = torch.from_numpy(np.asarray([left_classes[idx] for idx in indices[0]]))
                right_predicted_classes = torch.from_numpy(np.asarray([right_classes[idx] for idx in indices[1]]))
            left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
            right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
            left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
            right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())

        return (
            left_predicted_bboxes_refine,
            right_predicted_bboxes_refine,
            left_image_target_bboxes,
            right_image_target_bboxes,
        )
    
    def test_step_emd_ground_truth(self, emd_model, batch, batch_index, dataloader_index=0):
        left_predicted_bboxes = batch["left_image_target_bboxes"]
        right_predicted_bboxes = batch["right_image_target_bboxes"]
        left_image_target_bboxes = []
        right_image_target_bboxes = []
        left_predicted_bboxes_refine = []
        right_predicted_bboxes_refine = []
        for index in range(len(left_predicted_bboxes)):
            left_image_path = batch["left_image_path"][index]
            right_image_path = batch["right_image_path"][index]
            if left_image_path is not None:
                left_image = Image.open(left_image_path).convert('RGB')
            else:
                left_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["left_image"][0]))).convert('RGB')
            if right_image_path is not None:
                right_image = Image.open(right_image_path).convert('RGB')
            else:
                right_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["right_image"][0]))).convert('RGB')

            left_crop_bboxes = torch.cat([emd_transform(left_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in left_predicted_bboxes[index].detach().cpu().numpy()], dim=0)
            right_crop_bboxes = torch.cat([emd_transform(right_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in right_predicted_bboxes[index].detach().cpu().numpy()], dim=0)
            data = torch.cat((left_crop_bboxes, right_crop_bboxes)).to(device)

            emd_model.mode = 'encoder'
            data = emd_model(data)
            k = left_crop_bboxes.shape[0]
            data_shot, data_query = data[:k], data[k:]
            similarity_map = 1.0 - emd_model.get_similiarity_map(data_shot, data_query)
            similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
            indices = linear_sum_assignment(similarity_map)
            indices = [indices[1], indices[0]]

            left_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(left_predicted_bboxes[index][idx], idx)).unsqueeze(0) for idx in indices[0]], dim=0)
            right_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(right_predicted_bboxes[index][idx], idx)).unsqueeze(0) for idx in indices[1]], dim=0)
            left_predicted_classes = torch.zeros(len(indices[0]))
            right_predicted_classes = torch.zeros(len(indices[0]))
            left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
            right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
            left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
            right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())

        return (
            left_predicted_bboxes_refine,
            right_predicted_bboxes_refine,
            left_image_target_bboxes,
            right_image_target_bboxes,
        )
    
    def cosine_similarity(self, feature1, feature2):
        feature1_norm = feature1/(feature1.norm(dim=1) + 1e-8)[:, None]
        feature2_norm = feature2/(feature2.norm(dim=1) + + 1e-8)[:, None]
        cosine = torch.mm(feature1_norm, feature2_norm.transpose(0,1))

        return cosine

    def test_step_rcnn_ground_truth(self, feature_extractor, pooling, batch, batch_index, dataloader_index=0):
        left_predicted_bboxes = batch["left_image_target_bboxes"]
        right_predicted_bboxes = batch["right_image_target_bboxes"]
        left_image_target_bboxes = []
        right_image_target_bboxes = []
        left_predicted_bboxes_refine = []
        right_predicted_bboxes_refine = []
        for index in range(len(left_predicted_bboxes)):
            left_image_path = batch["left_image_path"][index]
            right_image_path = batch["right_image_path"][index]
            if left_image_path is not None:
                left_image = Image.open(left_image_path).convert('RGB')
            else:
                left_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["left_image"][0]))).convert('RGB')
            if right_image_path is not None:
                right_image = Image.open(right_image_path).convert('RGB')
            else:
                right_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["right_image"][0]))).convert('RGB')

            # Remove trivial (1 pixel) boxes
            left_bboxes_refine = left_predicted_bboxes[index].detach().cpu().numpy()
            right_bboxes_refine = right_predicted_bboxes[index].detach().cpu().numpy()
            left_bboxes_refine = [bbox for bbox in left_bboxes_refine
                                     if np.abs(bbox[0]-bbox[2]) >= 1 and np.abs(bbox[1] - bbox[3]) >= 1]
            right_bboxes_refine = [bbox for bbox in right_bboxes_refine
                                     if np.abs(bbox[0]-bbox[2]) >= 1 and np.abs(bbox[1] - bbox[3]) >= 1]
            
            if len(left_bboxes_refine) == 0 or len(right_bboxes_refine) == 0:
                left_predicted_rcnn_hungarian = torch.zeros((len(left_predicted_bboxes[index]), 5))
                left_predicted_rcnn_hungarian[:, 4] = torch.rand(len(left_predicted_bboxes[index]))
                left_predicted_classes = torch.zeros(left_predicted_rcnn_hungarian.shape[0])
                right_predicted_rcnn_hungarian = torch.zeros((len(right_predicted_bboxes[index]), 5))
                right_predicted_rcnn_hungarian[:, 4] = torch.rand(len(left_predicted_bboxes[index]))
                right_predicted_classes = torch.zeros(right_predicted_rcnn_hungarian.shape[0])

                left_predicted_bboxes_refine.append([left_predicted_rcnn_hungarian, left_predicted_classes])
                right_predicted_bboxes_refine.append([right_predicted_rcnn_hungarian, right_predicted_classes])
                left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
                right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())
                continue
                                                 
            left_crop_bbox_features = torch.cat([torch.flatten(pooling(feature_extractor(faster_rcnn_transform(left_image.crop(bbox[:4]*ratio)).unsqueeze(0).to(device))["backbone.body.layer4.2.relu"]), 1) for bbox in left_bboxes_refine], dim=0)
            right_crop_bbox_features = torch.cat([torch.flatten(pooling(feature_extractor(faster_rcnn_transform(right_image.crop(bbox[:4]*ratio)).unsqueeze(0).to(device))["backbone.body.layer4.2.relu"]), 1) for bbox in right_bboxes_refine], dim=0)

            similarity_map = 1.0 - self.cosine_similarity(left_crop_bbox_features, right_crop_bbox_features).detach().cpu().numpy()
            indices = linear_sum_assignment(similarity_map)

            left_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(left_predicted_bboxes[index][idx], idx)).unsqueeze(0) for idx in indices[0]], dim=0)
            right_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(right_predicted_bboxes[index][idx], idx)).unsqueeze(0) for idx in indices[1]], dim=0)
            left_predicted_classes = torch.zeros(len(indices[0]))
            right_predicted_classes = torch.zeros(len(indices[0]))
            left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
            right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
            left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
            right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())

        return (
            left_predicted_bboxes_refine,
            right_predicted_bboxes_refine,
            left_image_target_bboxes,
            right_image_target_bboxes,
        )

    def test_step_pre_process_and_emd_ground_truth(self, cyws_3d_model, depth_predictor, correspondence_extractor, 
                                                   emd_model, batch):
        left_predicted_bboxes = batch["left_image_target_bboxes"]
        right_predicted_bboxes = batch["right_image_target_bboxes"]
        left_image_target_bboxes = []
        right_image_target_bboxes = []
        left_predicted_bboxes_refine = []
        right_predicted_bboxes_refine = []

        for index in range(len(left_predicted_bboxes)):
            # Find the transformation matrix between left and right images
            batch_3d = create_batch_from_image_data([(batch["left_image"][0], batch["right_image"][0])])
            batch_3d = fill_in_the_missing_information(batch_3d, depth_predictor, correspondence_extractor)
            batch_3d = prepare_batch_for_model(batch_3d)
            M_1_to_2, M_2_to_1, transform_points_1_to_2, transform_points_2_to_1 = cyws_3d_model.registeration_module.transform_matrix(batch_3d)
            if torch.all(torch.eq(M_1_to_2[0], torch.eye(3).to(device))):
                left_image_path = batch["left_image_path"][index]
                right_image_path = batch["right_image_path"][index]
                if left_image_path is not None:
                    left_image = Image.open(left_image_path).convert('RGB')
                else:
                    left_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["left_image"][0]))).convert('RGB')
                if right_image_path is not None:
                    right_image = Image.open(right_image_path).convert('RGB')
                else:
                    right_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["right_image"][0]))).convert('RGB')

                left_crop_bboxes = torch.cat([emd_transform(left_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in left_predicted_bboxes[index].detach().cpu().numpy()], dim=0)
                right_crop_bboxes = torch.cat([emd_transform(right_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in right_predicted_bboxes[index].detach().cpu().numpy()], dim=0)
                data = torch.cat((left_crop_bboxes, right_crop_bboxes)).to(device)

                emd_model.mode = 'encoder'
                data = emd_model(data)
                k = left_crop_bboxes.shape[0]
                data_shot, data_query = data[:k], data[k:]
                similarity_map = 1.0 - emd_model.get_similiarity_map(data_shot, data_query)
                similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
                indices = linear_sum_assignment(similarity_map)
                indices = [indices[1], indices[0]]

                left_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(left_predicted_bboxes[index][idx], idx + 1)).unsqueeze(0) for idx in indices[0]], dim=0)
                right_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(right_predicted_bboxes[index][idx], idx + 1)).unsqueeze(0) for idx in indices[1]], dim=0)
                left_predicted_classes = torch.zeros(len(indices[0]))
                right_predicted_classes = torch.zeros(len(indices[0]))
                left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
                right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
                left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
                right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())
            else:
                # Tranform from left to right
                points1 = []
                for bbox in left_predicted_bboxes[index]:
                    points1.append([bbox[0], bbox[1]])
                    points1.append([bbox[2], bbox[3]])
                points1 = torch.from_numpy(np.asarray(points1)).to(device).reshape(-1, 2)/256.0
                points_1_to_2 = transform_points_1_to_2(points1, 0).reshape(-1, 4)*256.0
                points_1_to_2 = points_1_to_2.cpu()
                # Fix the situation where the transformation matrix estimated wrong
                if (points_1_to_2[:, 2:] >= points_1_to_2[:, :2]).all():
                    left_predicted_bboxes_tmp = []
                    right_predicted_bboxes_tmp = []
                    left_remove_index = []
                    right_remove_index = []
                    iou, union = box_ops.box_iou(points_1_to_2, right_predicted_bboxes[index][:,:4])
                    iou_matching_mask = iou > self.args.iou_threshold
                    iou_matching_mask = iou_matching_mask.long()
                    # Find maximum matching in each row
                    iou_max_row = torch.argmax(iou_matching_mask, dim=1)
                    iou_max_col = torch.argmax(iou_matching_mask, dim=0)
                    for i, idx in enumerate(iou_max_row):
                        if iou_matching_mask[i, idx] == False:
                            continue
                        else:
                            if iou_max_col[idx] == i:
                                left_predicted_bboxes_tmp.append(torch.from_numpy(np.append(left_predicted_bboxes[index][i].detach().cpu().numpy(), i + 1)).unsqueeze(0))
                                right_predicted_bboxes_tmp.append(torch.from_numpy(np.append(right_predicted_bboxes[index][idx].detach().cpu().numpy(), i + 1)).unsqueeze(0))
                                left_remove_index.append(i)
                                right_remove_index.append(idx)

                    left_keep_index = [i for i in range(len(left_predicted_bboxes[index])) if i not in left_remove_index]
                    right_keep_index = [i for i in range(len(right_predicted_bboxes[index])) if i not in right_remove_index]
                    
                    if len(left_keep_index) > 0  and len(right_keep_index) > 0:
                        # Hungarian matching for remaining boxes
                        left_image_path = batch["left_image_path"][index]
                        right_image_path = batch["right_image_path"][index]
                        if left_image_path is not None:
                            left_image = Image.open(left_image_path).convert('RGB')
                        else:
                            left_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["left_image"][0]))).convert('RGB')
                        if right_image_path is not None:
                            right_image = Image.open(right_image_path).convert('RGB')
                        else:
                            right_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["right_image"][0]))).convert('RGB')
                        left_crop_bboxes = torch.cat([emd_transform(left_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in left_predicted_bboxes[index][left_keep_index].detach().cpu().numpy()], dim=0)
                        right_crop_bboxes = torch.cat([emd_transform(right_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in right_predicted_bboxes[index][right_keep_index].detach().cpu().numpy()], dim=0)
                        data = torch.cat((left_crop_bboxes, right_crop_bboxes)).to(device)
                        emd_model.mode = 'encoder'
                        data = emd_model(data)
                        k = left_crop_bboxes.shape[0]
                        data_shot, data_query = data[:k], data[k:]
                        similarity_map = 1.0 - emd_model.get_similiarity_map(data_shot, data_query)
                        similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
                        indices = linear_sum_assignment(similarity_map)
                        indices = [indices[1], indices[0]]

                        for i, idx in enumerate(indices[0]):
                            left_predicted_bboxes_tmp.append(torch.from_numpy(np.append(left_predicted_bboxes[index][left_keep_index][idx].detach().cpu().numpy(), i + 1000)).unsqueeze(0))
                        for i, idx in enumerate(indices[1]):
                            right_predicted_bboxes_tmp.append(torch.from_numpy(np.append(right_predicted_bboxes[index][right_keep_index][idx].detach().cpu().numpy(), i + 1000)).unsqueeze(0))

                    left_predicted_emd_hungarian = torch.cat(left_predicted_bboxes_tmp, dim=0)
                    right_predicted_emd_hungarian = torch.cat(right_predicted_bboxes_tmp, dim=0)
                    left_predicted_classes = torch.zeros(left_predicted_emd_hungarian.shape[0])
                    right_predicted_classes = torch.zeros(left_predicted_emd_hungarian.shape[0])
                    left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
                    right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
                    left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
                    right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())
                else:
                    left_image_path = batch["left_image_path"][index]
                    right_image_path = batch["right_image_path"][index]
                    if left_image_path is not None:
                        left_image = Image.open(left_image_path).convert('RGB')
                    else:
                        left_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["left_image"][0]))).convert('RGB')
                    if right_image_path is not None:
                        right_image = Image.open(right_image_path).convert('RGB')
                    else:
                        right_image = Image.fromarray(np.uint8(K.tensor_to_image(batch["right_image"][0]))).convert('RGB')

                    left_crop_bboxes = torch.cat([emd_transform(left_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in left_predicted_bboxes[index].detach().cpu().numpy()], dim=0)
                    right_crop_bboxes = torch.cat([emd_transform(right_image.crop(bbox[:4]*ratio)).unsqueeze(0) for bbox in right_predicted_bboxes[index].detach().cpu().numpy()], dim=0)
                    data = torch.cat((left_crop_bboxes, right_crop_bboxes)).to(device)

                    emd_model.mode = 'encoder'
                    data = emd_model(data)
                    k = left_crop_bboxes.shape[0]
                    data_shot, data_query = data[:k], data[k:]
                    similarity_map = 1.0 - emd_model.get_similiarity_map(data_shot, data_query)
                    similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
                    indices = linear_sum_assignment(similarity_map)
                    indices = [indices[1], indices[0]]

                    left_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(left_predicted_bboxes[index][idx], idx + 1)).unsqueeze(0) for idx in indices[0]], dim=0)
                    right_predicted_emd_hungarian = torch.cat([torch.from_numpy(np.append(right_predicted_bboxes[index][idx], idx + 1)).unsqueeze(0) for idx in indices[1]], dim=0)
                    left_predicted_classes = torch.zeros(len(indices[0]))
                    right_predicted_classes = torch.zeros(len(indices[0]))
                    left_predicted_bboxes_refine.append([left_predicted_emd_hungarian, left_predicted_classes])
                    right_predicted_bboxes_refine.append([right_predicted_emd_hungarian, right_predicted_classes])
                    left_image_target_bboxes.append(batch["left_image_target_bboxes"][index].cpu())
                    right_image_target_bboxes.append(batch["right_image_target_bboxes"][index].cpu())

        return (
            left_predicted_bboxes_refine,
            right_predicted_bboxes_refine,
            left_image_target_bboxes,
            right_image_target_bboxes,
        )

    def test_epoch_end(self, multiple_test_set_outputs):
        """
        Test set evaluation function.
        """
        if len(self.test_set_names) == 1:
            multiple_test_set_outputs = [multiple_test_set_outputs]
        # iterate over all the test sets
        for test_set_name, test_set_batch_outputs in zip(
            self.test_set_names, multiple_test_set_outputs
        ):
            predicted_bboxes = []
            target_bboxes = []
            # iterate over all the batches for the current test set
            num_bboxes_predicted = []
            num_bboxes_target = []
            for test_set_outputs in test_set_batch_outputs:
                (
                    left_predicted_bboxes,
                    right_predicted_bboxes,
                    left_target_bboxes,
                    right_target_bboxes,
                ) = test_set_outputs
                # iterate over all predictions for images
                for bboxes_per_side in [left_predicted_bboxes, right_predicted_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        # filter out background bboxes
                        bboxes_per_image = bboxes_per_image[0][bboxes_per_image[1] == 0]
                        bboxes_per_image = bboxes_per_image[torch.where(bboxes_per_image[:, 4] >= self.args.det_thres)]
                        num_bboxes_predicted.append(len(bboxes_per_image))
                        bbox_list = BoxList(
                            bboxes_per_image[:, :4],
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("scores", bboxes_per_image[:, 4])
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        predicted_bboxes.append(bbox_list)
                # iterate over all targets for images
                for bboxes_per_side in [left_target_bboxes, right_target_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        num_bboxes_target.append(len(bboxes_per_image))
                        bbox_list = BoxList(
                            bboxes_per_image,
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        bbox_list.add_field("difficult", torch.zeros(bboxes_per_image.shape[0]))
                        target_bboxes.append(bbox_list)
            # compute metrics
            ap_map_precision_recall = eval_detection_voc(
                predicted_bboxes, target_bboxes, iou_thresh=0.5, use_07_metric=False
            )
            print('Length of precision is: {}'.format(len(ap_map_precision_recall['precision'][1])))
            if len(ap_map_precision_recall['precision'][1]) > 0:
                print('Final precision is: {}'.format(ap_map_precision_recall['precision'][1][-1]))
            else:
                print('Final precision is: {}'.format(0))
            print('Length of recall is: {}'.format(len(ap_map_precision_recall['recall'][1])))
            if len(ap_map_precision_recall['recall'][1]) > 0:
                print('Final recall is: {}'.format(ap_map_precision_recall['recall'][1][-1]))
            else:
                print('Final recall is: {}'.format(0))
            print('Num of predicted boxes after remove nosing: {}'.format(sum(num_bboxes_predicted)))
            print('Num of target boxes: {}'.format(sum(num_bboxes_target)))
            print('Num of predicted boxes images: {}'.format(len(num_bboxes_predicted)))
            print('Num of target boxes images: {}'.format(len(num_bboxes_target)))
            L.log(
                "INFO",
                f"{test_set_name} AP: {ap_map_precision_recall['ap']}, mAP: {ap_map_precision_recall['map']}")

    def test_epoch_visualize(self, multiple_test_set_outputs):
        """
        Test set evaluation function.
        """
        if len(self.test_set_names) == 1:
            multiple_test_set_outputs = [multiple_test_set_outputs]
        # iterate over all the test sets
        for test_set_name, test_set_batch_outputs in zip(
            self.test_set_names, multiple_test_set_outputs
        ):
            predicted_bboxes = []
            target_bboxes = []
            # iterate over all the batches for the current test set
            num_bboxes_predicted = []
            num_bboxes_target = []
            for test_set_outputs in test_set_batch_outputs:
                (
                    left_predicted_bboxes,
                    right_predicted_bboxes,
                    left_target_bboxes,
                    right_target_bboxes,
                ) = test_set_outputs
                # iterate over all predictions for images
                for bboxes_per_side in [left_predicted_bboxes, right_predicted_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        # filter out background bboxes
                        bboxes_per_image = bboxes_per_image[0][bboxes_per_image[1] == 0]
                        num_bboxes_predicted.append(len(bboxes_per_image))
                        bbox_list = BoxList(
                            bboxes_per_image[:, :4],
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("scores", bboxes_per_image[:, 4])
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        predicted_bboxes.append(bbox_list)
                # iterate over all targets for images
                for bboxes_per_side in [left_target_bboxes, right_target_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        num_bboxes_target.append(len(bboxes_per_image))
                        bbox_list = BoxList(
                            bboxes_per_image,
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        bbox_list.add_field("difficult", torch.zeros(bboxes_per_image.shape[0]))
                        target_bboxes.append(bbox_list)
            # compute metrics
            ap_map_precision_recall = eval_detection_voc(
                predicted_bboxes, target_bboxes, iou_thresh=0.5, use_07_metric=False
            )

            # disp = PrecisionRecallDisplay(precision=ap_map_precision_recall['precision'][1], recall=ap_map_precision_recall['recall'][1])
            # disp.plot()
            # plt.savefig(os.path.join(self.args.dst, test_set_name + '.png'))

            os.makedirs(os.path.join(self.args.dst, test_set_name), exist_ok=True)
            np.save(os.path.join(self.args.dst, test_set_name, self.args.name + '_precision.npy'), np.asarray(ap_map_precision_recall['precision'][1]))
            np.save(os.path.join(self.args.dst, test_set_name, self.args.name + '_recall.npy'), np.asarray(ap_map_precision_recall['recall'][1]))

            print('Length of precision is: {}'.format(len(ap_map_precision_recall['precision'][1])))
            print('Final precision is: {}'.format(ap_map_precision_recall['precision'][1][-1]))
            print('Length of recall is: {}'.format(len(ap_map_precision_recall['recall'][1])))
            print('Final recall is: {}'.format(ap_map_precision_recall['recall'][1][-1]))
            print('Num of predicted boxes after remove nosing: {}'.format(sum(num_bboxes_predicted)))
            print('Num of target boxes: {}'.format(sum(num_bboxes_target)))
            print('Num of predicted boxes images: {}'.format(len(num_bboxes_predicted)))
            print('Num of target boxes images: {}'.format(len(num_bboxes_target)))
            L.log(
                "INFO",
                f"{test_set_name} AP: {ap_map_precision_recall['ap']}, mAP: {ap_map_precision_recall['map']}")

    def configure_optimizers(self):
        optimizer_params = [
            {"params": [parameter for parameter in self.parameters() if parameter.requires_grad]}
        ]
        optimizer = torch.optim.Adam(optimizer_params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, batch):
        left_image_encoded_features = self.unet_model.encoder(batch["left_image"].to(device))
        right_image_encoded_features = self.unet_model.encoder(batch["right_image"].to(device))
        left_image_encoded_features_last = left_image_encoded_features[-1]
        right_image_encoded_features_last = right_image_encoded_features[-1]
        for i in range(len(self.coattention_modules)):
            (
                left_image_encoded_features[-(i + 1)],
                right_image_encoded_features[-(i + 1)],
            ) = self.coattention_modules[i](
                left_image_encoded_features[-(i + 1)], right_image_encoded_features[-(i + 1)]
            )
        left_image_decoded_features = self.unet_model.decoder(*left_image_encoded_features)
        right_image_decoded_features = self.unet_model.decoder(*right_image_encoded_features)
        return (
            self.centernet_head([left_image_decoded_features]),
            self.centernet_head([right_image_decoded_features]),
            left_image_encoded_features_last,
            right_image_encoded_features_last
        )
    

def suppress_non_maximum(bboxes):
    maximum = np.ones(len(bboxes))
    results = []
    for i in range(len(bboxes)):
        if not maximum[i]:
            continue
        results.append(bboxes[i])
        for j in range(i + 1, len(bboxes)):
            left_object = shapely.geometry.box(*bboxes[i])
            right_object = shapely.geometry.box(*bboxes[j])
            if right_object.area == 0:
                maximum[j] = 0
                continue
            iou = left_object.intersection(right_object).area / left_object.union(right_object).area
            if iou > 0.5:
                maximum[j] = 0
            if right_object.intersection(left_object).area / right_object.area > 0.5:
                maximum[j] = 0
    return results, maximum

def marshal_getitem_data(data, split):
    """
    The data field above is returned by the individual datasets.
    This function marshals that data into the format expected by this
    model/method.
    """
    if split in ["train", "val", "test"]:
        (
            data["image1"],
            target_region_and_annotations,
        ) = utils_cyws.geometry.resize_image_and_annotations(
            data["image1"],
            output_shape_as_hw=(256, 256),
            annotations=data["image1_target_annotations"],
        )
        data["image1_target_annotations"] = target_region_and_annotations
        (
            data["image2"],
            target_region_and_annotations,
        ) = utils_cyws.geometry.resize_image_and_annotations(
            data["image2"],
            output_shape_as_hw=(256, 256),
            annotations=data["image2_target_annotations"],
        )
        data["image2_target_annotations"] = target_region_and_annotations

    assert data["image1"].shape == data["image2"].shape
    image1_target_bboxes = torch.Tensor([x["bbox"] for x in data["image1_target_annotations"]])
    image2_target_bboxes = torch.Tensor([x["bbox"] for x in data["image2_target_annotations"]])

    if len(image1_target_bboxes) != len(image2_target_bboxes) or len(image1_target_bboxes) == 0:
        return None

    return {
        "left_image": data["image1"],
        "right_image": data["image2"],
        "left_image_path": data["image1_path"],
        "right_image_path": data["image2_path"],
        "left_image_target_bboxes": image1_target_bboxes,
        "right_image_target_bboxes": image2_target_bboxes,
        "target_bbox_labels": torch.zeros(len(image1_target_bboxes)).long(),
        "query_metadata": {
            "pad_shape": data["image1"].shape[-2:],
            "border": np.array([0, 0, 0, 0]),
            "batch_input_shape": data["image1"].shape[-2:],
        },
    }


def dataloader_collate_fn(batch):
    """
    Defines the collate function for the dataloader specific to this
    method/model.
    """
    if batch[0] is None:
        return None

    keys = batch[0].keys()
    collated_dictionary = {}
    for key in keys:
        collated_dictionary[key] = []
        for batch_item in batch:
            collated_dictionary[key].append(batch_item[key])
        if key in [
            "left_image_path",
            "right_image_path",
            "left_image_target_bboxes",
            "right_image_target_bboxes",
            "query_metadata",
            "target_bbox_labels",
        ]:
            continue
        collated_dictionary[key] = ImageList.from_tensors(
            collated_dictionary[key], size_divisibility=32
        ).tensor

    return collated_dictionary


################################################
## The callback manager below handles logging ##
## to Weights And Biases.                     ##
################################################


class WandbCallbackManager(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        datamodule = DataModule(args)
        datamodule.setup()
        self.test_set_names = datamodule.test_dataset_names

    @rank_zero_only
    def on_fit_start(self, trainer, model):
        if self.args.no_logging:
            return
        trainer.logger.experiment.config.update(self.args, allow_val_change=True)

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, model, predicted_bboxes, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            self.val_batch = batch
            left_predicted_bboxes, right_predicted_bboxes = predicted_bboxes
            self.val_set_predicted_bboxes = [
                (
                    predicted_bboxes_per_left_image[0][predicted_bboxes_per_left_image[1] == 0].to(
                        "cpu"
                    ),
                    (
                        predicted_bboxes_per_right_image[0][
                            predicted_bboxes_per_right_image[1] == 0
                        ].to("cpu")
                    ),
                )
                for predicted_bboxes_per_left_image, predicted_bboxes_per_right_image in zip(
                    left_predicted_bboxes, right_predicted_bboxes
                )
            ]
            self.val_set_target_bboxes = [
                (target_bboxes_per_left_image.to("cpu"), target_bboxes_per_right_image.to("cpu"))
                for target_bboxes_per_left_image, target_bboxes_per_right_image in zip(
                    batch["left_image_target_bboxes"], batch["right_image_target_bboxes"]
                )
            ]

    @rank_zero_only
    def on_validation_end(self, trainer, model):
        self.log_qualitative_predictions(
            self.val_batch,
            self.val_set_predicted_bboxes,
            self.val_set_target_bboxes,
            "val",
            trainer,
        )

    @rank_zero_only
    def on_test_start(self, trainer, model):
        self.test_batches = [[] for _ in range(len(self.test_set_names))]
        self.test_set_predicted_bboxes = [[] for _ in range(len(self.test_set_names))]
        self.test_set_target_bboxes = [[] for _ in range(len(self.test_set_names))]

    @rank_zero_only
    def on_test_batch_end(
        self, trainer, model, predicted_and_target_bboxes, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx == 0:
            self.test_batches[dataloader_idx] = batch
            (
                left_predicted_bboxes,
                right_predicted_bboxes,
                left_target_bboxes,
                right_target_bboxes,
            ) = predicted_and_target_bboxes
            self.test_set_predicted_bboxes[dataloader_idx] = [
                (
                    predicted_bboxes_per_left_image[0][predicted_bboxes_per_left_image[1] == 0],
                    (predicted_bboxes_per_right_image[0][predicted_bboxes_per_right_image[1] == 0]),
                )
                for predicted_bboxes_per_left_image, predicted_bboxes_per_right_image in zip(
                    left_predicted_bboxes, right_predicted_bboxes
                )
            ]
            self.test_set_target_bboxes[dataloader_idx] = [
                (target_bboxes_per_left_image, target_bboxes_per_right_image)
                for target_bboxes_per_left_image, target_bboxes_per_right_image in zip(
                    left_target_bboxes, right_target_bboxes
                )
            ]

    @rank_zero_only
    def on_test_end(self, trainer, model):
        for test_set_index, test_set_name in enumerate(self.test_set_names):
            self.log_qualitative_predictions(
                self.test_batches[test_set_index],
                self.test_set_predicted_bboxes[test_set_index],
                self.test_set_target_bboxes[test_set_index],
                f"test_{test_set_name}",
                trainer,
            )

    def log_qualitative_predictions(
        self,
        batch,
        predicted_bboxes,
        target_bboxes,
        batch_name,
        trainer,
    ):
        """
        Logs the predicted masks for a single val/test batch for qualitative analysis.
        """
        outputs = []
        for (
            left_image,
            right_image,
            target_bboxes_per_image,
            predicted_bboxes_per_image,
        ) in zip(batch["left_image"], batch["right_image"], target_bboxes, predicted_bboxes):
            for this_image, predicted_bboxes_per_image, target_bboxes_per_image in zip(
                [left_image, right_image], predicted_bboxes_per_image, target_bboxes_per_image
            ):
                predicted_bboxes_for_this_image = self.get_wandb_bboxes(
                    predicted_bboxes_per_image, class_id=1
                )
                ground_truth_bboxes_for_this_image = self.get_wandb_bboxes(
                    target_bboxes_per_image, class_id=2
                )
                outputs.append(
                    wandb.Image(
                        K.tensor_to_image(this_image),
                        boxes={
                            "predictions": {"box_data": predicted_bboxes_for_this_image},
                            "ground_truth": {"box_data": ground_truth_bboxes_for_this_image},
                        },
                    )
                )
        L.log("INFO", f"Finished computing qualitative predictions for {batch_name}.")
        if not self.args.no_logging:
            trainer.logger.experiment.log(
                {
                    f"qualitative_predictions/{batch_name}": outputs,
                    "global_step": trainer.global_step,
                }
            )

    def get_wandb_bboxes(self, bboxes_per_image, class_id):
        boxes_for_this_image = []
        image_width, image_height = 256, 256
        try:
            scores = bboxes_per_image[:, 4]
        except:
            scores = None
        for index, box in enumerate(bboxes_per_image[:, :4]):
            x1, y1 = [box[0].item() / image_width, box[1].item() / image_height]
            x2, y2 = [box[2].item() / image_width, box[3].item() / image_height]
            this_box_data = {
                "position": {"minX": x1, "maxX": x2, "minY": y1, "maxY": y2},
                "class_id": class_id,
            }
            if scores is not None:
                this_box_data.update({"scores": {"score": scores[index].item()}})
            boxes_for_this_image.append(this_box_data)
        return boxes_for_this_image
