import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import matplotlib.patches as patches
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as transforms
import shapely.geometry
from .box_ops import box_iou
import kornia as K

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_batch_from_image_data, create_batch_from_image_path, fill_in_the_missing_information, prepare_batch_for_model


# 'CPU' or 'CUDA'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

color_map = {'0': 'green',
             '1': 'red',
             '2': 'orange',
             '3': 'yellow',
             '4': 'pink',
             '5': 'purple',
             '6': 'brown',
             '7': 'gray',
             '8': 'cyan',
             '9': 'coral',
             '10': 'blue',
             '11': 'wheat',
             '12': 'coral',
             '13': 'yellowgreen',
             '14': 'mediumslateblue'}

# EMD transform
image_size = 84
emd_transform = transforms.Compose([
    transforms.Resize([92, 92]),
    transforms.CenterCrop(image_size),

    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

cyws_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


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
            iou = left_object.intersection(right_object).area /(left_object.union(right_object).area + 1e-10)
            if iou > 0.5:
                maximum[j] = 0
            if right_object.intersection(left_object).area /(right_object.area + 1e-10) > 0.5:
                maximum[j] = 0
    return results


def crop_image(img, bbox, ratio=np.asarray([1.0, 1.0, 1.0, 1.0]), img_size=(256, 256)):
    center_x = (bbox[0] + bbox[2])/2
    center_y = (bbox[1] + bbox[3])/2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    ori_crop_box = ratio*bbox
    top_left_box = ratio*np.asarray([max(center_x-w, 0), max(center_y-h, 0), center_x, center_y])
    top_right_box = ratio*np.asarray([center_x, max(center_y-h, 0), min(center_x+w, img_size[0]), center_y])
    bot_left_box = ratio*np.asarray([max(center_x-w, 0), center_y, center_x, min(center_y+h, img_size[1])])
    bot_right_box = ratio*np.asarray([center_x, center_y, min(center_x+w, img_size[0]), min(center_y+h, img_size[1])])

    ori_crop = img.crop(ori_crop_box)
    top_left_crop = img.crop(top_left_box)
    top_right_crop = img.crop(top_right_box)
    bot_left_crop = img.crop(bot_left_box)
    bot_right_crop = img.crop(bot_right_box)

    return [ori_crop, top_left_crop, top_right_crop, bot_left_crop, bot_right_crop]


def post_processing(args, cyws_3d_model, depth_predictor, correspondence_extractor, data1, data2, 
                    left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                    subplots, name, use_path=True):
    # Create folder to save the results
    os.makedirs(args.dst, exist_ok=True)

    if use_path:
        batch = create_batch_from_image_path([(data1, data2)])
    else:
        batch = create_batch_from_image_data([(data1, data2)])
    batch = fill_in_the_missing_information(batch, depth_predictor, correspondence_extractor)
    batch = prepare_batch_for_model(batch)
    M_1_to_2, M_2_to_1, transform_points_1_to_2, transform_points_2_to_1 = cyws_3d_model.registeration_module.transform_matrix(batch)
    if torch.all(torch.eq(M_1_to_2[0], torch.eye(3).to(device))):
        left_predicted_bboxes = np.asarray(left_predicted_bboxes)
        right_predicted_bboxes = np.asarray(right_predicted_bboxes)
        left_predicted_bboxes = left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
        right_predicted_bboxes = right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
        left_predicted_classes = torch.zeros(len(left_predicted_bboxes), dtype=torch.long)
        right_predicted_classes = torch.zeros(len(right_predicted_bboxes), dtype=torch.long)
        return ([[torch.from_numpy(left_predicted_bboxes.copy()).cpu(), left_predicted_classes.cpu()]],
                [[torch.from_numpy(right_predicted_bboxes.copy()).cpu(), right_predicted_classes.cpu()]],
                [left_target_bboxes.cpu()],
                [right_target_bboxes.cpu()])

    if args.remove_noise:
        # Tranform from left to right
        points1 = []
        for bbox in left_predicted_bboxes:
            points1.append([bbox[0], bbox[1]])
            points1.append([bbox[2], bbox[3]])
        points1 = torch.from_numpy(np.asarray(points1)).to(device).reshape(-1, 2)/256.0
        points_1_to_2 = transform_points_1_to_2(points1, 0).reshape(-1, 4)*256.0
        points_1_to_2 = points_1_to_2.cpu()
        # Fix the situation where the transformation matrix estimated wrong
        for idx in range(len(points_1_to_2)):
            if points_1_to_2[idx, 2] < points_1_to_2[idx, 0]:
                left_predicted_bboxes = np.asarray(left_predicted_bboxes)
                right_predicted_bboxes = np.asarray(right_predicted_bboxes)
                left_predicted_bboxes = left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
                right_predicted_bboxes = right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
                left_predicted_classes = torch.zeros(len(left_predicted_bboxes), dtype=torch.long)
                right_predicted_classes = torch.zeros(len(right_predicted_bboxes), dtype=torch.long)
                return ([[torch.from_numpy(left_predicted_bboxes.copy()).cpu(), left_predicted_classes.cpu()]],
                        [[torch.from_numpy(right_predicted_bboxes.copy()).cpu(), right_predicted_classes.cpu()]],
                        [left_target_bboxes.cpu()],
                        [right_target_bboxes.cpu()])
            if points_1_to_2[idx, 3] < points_1_to_2[idx, 1]:
                left_predicted_bboxes = np.asarray(left_predicted_bboxes)
                right_predicted_bboxes = np.asarray(right_predicted_bboxes)
                left_predicted_bboxes = left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
                right_predicted_bboxes = right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
                left_predicted_classes = torch.zeros(len(left_predicted_bboxes), dtype=torch.long)
                right_predicted_classes = torch.zeros(len(right_predicted_bboxes), dtype=torch.long)
                return ([[torch.from_numpy(left_predicted_bboxes.copy()).cpu(), left_predicted_classes.cpu()]],
                        [[torch.from_numpy(right_predicted_bboxes.copy()).cpu(), right_predicted_classes.cpu()]],
                        [left_target_bboxes.cpu()],
                        [right_target_bboxes.cpu()])
        left_predicted_bboxes = torch.from_numpy(np.asarray(left_predicted_bboxes))
        right_predicted_bboxes = torch.from_numpy(np.asarray(right_predicted_bboxes))
        iou, union = box_iou(points_1_to_2, right_predicted_bboxes[:,:4])
        iou_matching = iou > args.iou_threshold
        iou_matching = iou_matching.long()
        left_boxes_idxs = torch.where(torch.sum(iou_matching, dim=1) > 0)[0]   # Find all noise boxes in the left image
        right_boxes_idxs = torch.where(torch.sum(iou_matching, dim=0) > 0)[0]   # Find all noise boxes in the right image
        left_predicted_bboxes = left_predicted_bboxes[left_boxes_idxs]
        right_predicted_bboxes = right_predicted_bboxes[right_boxes_idxs]

    if args.visualize:
        left_predicted_bboxes = left_predicted_bboxes.cpu().numpy().tolist()
        right_predicted_bboxes = right_predicted_bboxes.cpu().numpy().tolist()
        left_target_bboxes = left_target_bboxes.cpu().numpy()
        right_target_bboxes = right_target_bboxes.cpu().numpy()

        # Draw the bounding boxes for left and right images after post processing
        for index, bboxes in zip([0, 1], [left_predicted_bboxes, right_predicted_bboxes]):
            for bbox in bboxes:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor=color_map['2'], facecolor="none"
                )
                subplots[index].add_patch(rect)

        # Draw the ground-truth
        for index, bboxes in zip([0, 1], [left_target_bboxes, right_target_bboxes]):
            if len(bboxes) == 0:
                continue
            for bbox in bboxes:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor=color_map['0'], facecolor="none", linestyle='dashed'
                )
                subplots[index].add_patch(rect)

        for subplot in subplots:
            subplot.axis("off")
        plt.savefig(os.path.join(args.dst, name)) 
    else:
        left_predicted_bboxes = np.asarray(left_predicted_bboxes)
        right_predicted_bboxes = np.asarray(right_predicted_bboxes)
        left_predicted_bboxes = left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
        right_predicted_bboxes = right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
        left_predicted_classes = torch.zeros(len(left_predicted_bboxes), dtype=torch.long)
        right_predicted_classes = torch.zeros(len(right_predicted_bboxes), dtype=torch.long)

        return ([[torch.from_numpy(left_predicted_bboxes.copy()).cpu(), left_predicted_classes.cpu()]],
                [[torch.from_numpy(right_predicted_bboxes.copy()).cpu(), right_predicted_classes.cpu()]],
                [left_target_bboxes.cpu()],
                [right_target_bboxes.cpu()])


def cyws_3d(args, cyws_3d_model, depth_predictor, correspondence_extractor, data1, data2,
            left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes,
            subplots, name, use_path=True):
    # Creating folder to save the results
    os.makedirs(args.dst, exist_ok=True)

    # Remove confidence score 
    left_predicted_bboxes = left_predicted_bboxes[:, :4]
    right_predicted_bboxes = right_predicted_bboxes[:, :4]

    left_target_bboxes = left_target_bboxes.cpu().numpy()
    right_target_bboxes = right_target_bboxes.cpu().numpy()

    if use_path:
        batch = create_batch_from_image_path([(data1, data2)])
    else:
        batch = create_batch_from_image_data([(data1, data2)])
    batch = fill_in_the_missing_information(batch, depth_predictor, correspondence_extractor)
    batch = prepare_batch_for_model(batch)
    M_1_to_2, M_2_to_1, transform_points_1_to_2, transform_points_2_to_1 = cyws_3d_model.registeration_module.transform_matrix(batch)

    if args.left_to_right:
        # Draw left to right
        if args.use_ground_truth:
            points1 = []
            for bbox in left_target_bboxes:
                points1.append([bbox[0], bbox[1]])
                points1.append([bbox[2], bbox[3]])
            points1 = torch.from_numpy(np.asarray(points1)).to(device).reshape(-1, 2)/256.0
            points_1_to_2 = transform_points_1_to_2(points1, 0).cpu().numpy()*256.0

            for i, bbox in enumerate(left_target_bboxes):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor=color_map[str(i)], facecolor="none"
                )
                subplots[0].add_patch(rect)

            for i in range(len(points_1_to_2)//2):
                w = points_1_to_2[i*2+1][0] - points_1_to_2[i*2][0]
                h = points_1_to_2[i*2+1][1] - points_1_to_2[i*2][1]
                rect = patches.Rectangle(
                        (points_1_to_2[i*2][0], points_1_to_2[i*2][1]), w, h, linewidth=1, edgecolor=color_map[str(i)], facecolor="none"
                    )
                subplots[1].add_patch(rect)

            # Draw ground-truth for right image
            for i, bbox in enumerate(right_target_bboxes):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor=color_map[str(i)], facecolor="none", linestyle='dashed'
                )
                subplots[1].add_patch(rect)

            for subplot in subplots:
                subplot.axis("off")
            plt.savefig(os.path.join(args.dst, name))    
        else:
            points1 = []
            for bbox in left_predicted_bboxes:
                points1.append([bbox[0], bbox[1]])
                points1.append([bbox[2], bbox[3]])
            points1 = torch.from_numpy(np.asarray(points1)).to(device).reshape(-1, 2)/256.0
            points_1_to_2 = transform_points_1_to_2(points1, 0).cpu().numpy()*256.0

            # Draw for left predict boxes
            for i, bbox in enumerate(left_predicted_bboxes):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor=color_map['2'], facecolor="none"
                )
                subplots[0].add_patch(rect)

            # Draw for corresponding boxes from left image in right image
            for i in range(len(points_1_to_2)//2):
                w = points_1_to_2[i*2+1][0] - points_1_to_2[i*2][0]
                h = points_1_to_2[i*2+1][1] - points_1_to_2[i*2][1]
                rect = patches.Rectangle(
                        (points_1_to_2[i*2][0], points_1_to_2[i*2][1]), w, h, linewidth=1, edgecolor=color_map['0'], facecolor="none"
                    )
                subplots[1].add_patch(rect)

            # Draw for right predict boxes
            for i, bbox in enumerate(right_predicted_bboxes):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor=color_map['2'], facecolor="none"
                )
                subplots[1].add_patch(rect)

            for subplot in subplots:
                subplot.axis("off")
            plt.savefig(os.path.join(args.dst, name))     
    else:   
        # Draw right_to_left
        if args.use_ground_truth:
            points2 = []
            for bbox in right_target_bboxes:
                points2.append([bbox[0], bbox[1]])
                points2.append([bbox[2], bbox[3]])
            points2 = torch.from_numpy(np.asarray(points2)).to(device).reshape(-1, 2)/256.0
            points_2_to_1 = transform_points_2_to_1(points2, 0).cpu().numpy()*256.0

            for i, bbox in enumerate(right_target_bboxes):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    (points_1_to_2[i*2][0], points_1_to_2[i*2][1]), w, h, linewidth=1, edgecolor=color_map[str(i)], facecolor="none"
                )
                subplots[1].add_patch(rect)

            for i in range(len(points_2_to_1)//2):
                w = points_2_to_1[i*2+1][0] - points_2_to_1[i*2][0]
                h = points_2_to_1[i*2+1][1] - points_2_to_1[i*2][1]
                rect = patches.Rectangle(
                        (points_2_to_1[i*2][0], points_2_to_1[i*2][1]), w, h, linewidth=1, edgecolor=color_map[str(i)], facecolor="none"
                    )
                subplots[0].add_patch(rect)

            for subplot in subplots:
                subplot.axis("off")
            plt.savefig(os.path.join(args.dst, name))  
        else:
            points2 = []
            for bbox in right_predicted_bboxes:
                points2.append([bbox[0], bbox[1]])
                points2.append([bbox[2], bbox[3]])
            points2 = torch.from_numpy(np.asarray(points2)).to(device).reshape(-1, 2)/256.0
            points_2_to_1 = transform_points_2_to_1(points2, 0).cpu().numpy()*256.0

            for i, bbox in enumerate(right_predicted_bboxes):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=1, edgecolor=color_map[str(i)], facecolor="none"
                )
                subplots[1].add_patch(rect)

            for i in range(len(points_2_to_1)//2):
                w = points_2_to_1[i*2+1][0] - points_2_to_1[i*2][0]
                h = points_2_to_1[i*2+1][1] - points_2_to_1[i*2][1]
                rect = patches.Rectangle(
                        (points_2_to_1[i*2][0], points_2_to_1[i*2][1]), w, h, linewidth=1, edgecolor=color_map[str(i)], facecolor="none"
                    )
                subplots[0].add_patch(rect)

            for subplot in subplots:
                subplot.axis("off")
        plt.savefig(os.path.join(args.dst, name))  


def emd_hungarian(args, emd_model, left_predicted_bboxes, right_predicted_bboxes, 
                  left_target_bboxes, right_target_bboxes, 
                  data1, data2, subplots, name, use_path=True):
    # Creating folder to save the results
    os.makedirs(args.dst, exist_ok=True)

    # Extending bounding box ratio
    ratio = args.ratio

    # Remove confidence score 
    left_predicted_bboxes = left_predicted_bboxes[:, :4]
    right_predicted_bboxes = right_predicted_bboxes[:, :4]

    if use_path:
        left_img = Image.open(data1).convert('RGB')
        right_img = Image.open(data2).convert('RGB')
    else:
        left_img = data1
        right_img = data2

    left_bboxs = torch.cat([emd_transform(left_img.crop(bbox*ratio)).unsqueeze(0) for bbox in left_predicted_bboxes], dim=0)
    right_bboxs = torch.cat([emd_transform(right_img.crop(bbox*ratio)).unsqueeze(0) for bbox in right_predicted_bboxes], dim=0)
    data = torch.cat((left_bboxs, right_bboxs))
    emd_model.mode = 'encoder'
    data = emd_model(data)
    k = left_bboxs.shape[0]
    data_shot, data_query = data[:k], data[k:]  # shot: 5,3,84,84  query:75,3,84,84
    similarity_map = 1.0 - emd_model.get_similiarity_map(data_shot, data_query)
    similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
    indices = linear_sum_assignment(similarity_map)
    indices = [indices[1], indices[0]]

    # Draw the results
    for index, idxs, bboxes in zip([0, 1], indices, [left_predicted_bboxes, right_predicted_bboxes]):
        for idx in range(len(idxs)):
            bbox = bboxes[idxs[idx]]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle(
                bbox[:2], w, h, linewidth=1, edgecolor=color_map[str(idx)], facecolor="none"
            )
            subplots[index].add_patch(rect)

    # Draw the ground-truth
    for index, bboxes in zip([0, 1], [left_target_bboxes, right_target_bboxes]):
        if len(bboxes) == 0:
            continue
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle(
                bbox[:2], w, h, linewidth=1, edgecolor=color_map['5'], facecolor="none", linestyle='dashed'
            )
            subplots[index].add_patch(rect)

    for subplot in subplots:
        subplot.axis("off")

    plt.savefig(os.path.join(args.dst, name))


def wacv(args, left_predicted_bboxes, right_predicted_bboxes, 
                  left_target_bboxes, right_target_bboxes, subplots, name):
    os.makedirs(args.dst, exist_ok=True)

    # Draw the ground-truth
    for index, bboxes in zip([0, 1], [left_target_bboxes, right_target_bboxes]):
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle(
                bbox[:2], w, h, linewidth=2, edgecolor=color_map['10'], facecolor="none", linestyle='dashed'
            )
            subplots[index].add_patch(rect)

    # Draw the results
    for index, bboxes in zip([0, 1], [left_predicted_bboxes, right_predicted_bboxes]):
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle(
                bbox[:2], w, h, linewidth=2, edgecolor=color_map['2'], facecolor="none"
            )
            subplots[index].add_patch(rect)

    for subplot in subplots:
        subplot.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(args.dst, name.split('.')[0] + '.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)


def change_box_detector(args, diff_model, batch_with_single_item):
    if batch_with_single_item['left_image_target_bboxes'] is None:
        left_target_bboxes = []
    else:
        left_target_bboxes = batch_with_single_item['left_image_target_bboxes'][0]
    if batch_with_single_item['right_image_target_bboxes'] is None:
        right_target_bboxes = []
    else:
        right_target_bboxes = batch_with_single_item['right_image_target_bboxes'][0]

    left_predicted_bboxes, right_predicted_bboxes, _, _, left_image_encoded_features_last, right_image_encoded_features_last = diff_model.test_step(
        batch_with_single_item, 0
    )
    
    if args.use_nms:
        left_predicted_bboxes = left_predicted_bboxes[0][0].detach().numpy()
        right_predicted_bboxes = right_predicted_bboxes[0][0].detach().numpy()
        left_predicted_bboxes = suppress_non_maximum(
            left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.top_n_bboxes, :]
        )
        right_predicted_bboxes = suppress_non_maximum(
            right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.top_n_bboxes, :]
        )
        return left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last
    else:
        return left_predicted_bboxes[0][0].detach(), right_predicted_bboxes[0][0].detach(), left_target_bboxes, right_target_bboxes, left_image_encoded_features_last, right_image_encoded_features_last
    

# EMD-Hungarian (No visualization)
def emd_hungarian_matching(args, emd_model, data1, data2, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, use_path=False):
    if use_path:
        left_image = Image.open(data1).convert('RGB')
        right_image = Image.open(data2).convert('RGB')
    else:
        left_image = Image.fromarray(np.uint8(K.tensor_to_image(data1))).convert('RGB')
        right_image = Image.fromarray(np.uint8(K.tensor_to_image(data2))).convert('RGB')

    # Select boxes with the most confidence
    left_predicted_bboxes = np.asarray(left_predicted_bboxes)
    left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 4] > args.det_thres)[0]]
    right_predicted_bboxes = np.asarray(right_predicted_bboxes)
    right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 4] > args.det_thres)[0]]
    left_predicted_bboxes = left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
    right_predicted_bboxes = right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
    left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 2] > left_predicted_bboxes[:, 0])[0]]
    left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 3] > left_predicted_bboxes[:, 1])[0]]
    right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 2] > right_predicted_bboxes[:, 0])[0]]
    right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 3] > right_predicted_bboxes[:, 1])[0]]

    if len(left_predicted_bboxes) == 0 or len(right_predicted_bboxes) == 0:
        left_predicted_bboxes = torch.from_numpy(left_predicted_bboxes)
        right_predicted_bboxes = torch.from_numpy(right_predicted_bboxes)
        left_predicted_classes = torch.zeros(left_predicted_bboxes.shape[0])
        right_predicted_classes = torch.zeros(right_predicted_bboxes.shape[0])

        return ([[left_predicted_bboxes.cpu(), left_predicted_classes.cpu()]],
        [[right_predicted_bboxes.cpu(), right_predicted_classes.cpu()]],
        [left_target_bboxes.cpu()],
        [right_target_bboxes.cpu()])

    # EMD Hungarian algorithm
    left_crop_bboxes = torch.cat([emd_transform(left_image.crop(bbox[:4]*args.ratio)).unsqueeze(0) for bbox in left_predicted_bboxes], dim=0)
    right_crop_bboxes = torch.cat([emd_transform(right_image.crop(bbox[:4]*args.ratio)).unsqueeze(0) for bbox in right_predicted_bboxes], dim=0)
    data = torch.cat((left_crop_bboxes, right_crop_bboxes)).to(device)

    emd_model.mode = 'encoder'
    data = emd_model(data)
    k = left_crop_bboxes.shape[0]
    data_shot, data_query = data[:k], data[k:]
    similarity_map = 1.0 - emd_model.get_similiarity_map(data_shot, data_query)
    similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
    indices = linear_sum_assignment(similarity_map)
    indices = [indices[1], indices[0]]

    left_predicted_emd_hungarian = torch.cat([torch.from_numpy(left_predicted_bboxes[idx].copy()).unsqueeze(0) for idx in indices[0]], dim=0)
    right_predicted_emd_hungarian = torch.cat([torch.from_numpy(right_predicted_bboxes[idx].copy()).unsqueeze(0) for idx in indices[1]], dim=0)
    left_predicted_classes = torch.zeros(len(indices[0]))
    right_predicted_classes = torch.zeros(len(indices[0]))

    return ([[left_predicted_emd_hungarian.cpu(), left_predicted_classes.cpu()]],
        [[right_predicted_emd_hungarian.cpu(), right_predicted_classes.cpu()]],
        [left_target_bboxes.cpu()],
        [right_target_bboxes.cpu()])


def cosine_similarity(feature1, feature2):
    feature1_norm = feature1/(feature1.norm(dim=1) + 1e-8)[:, None]
    feature2_norm = feature2/(feature2.norm(dim=1) + + 1e-8)[:, None]
    cosine = torch.mm(feature1_norm, feature2_norm.transpose(0,1))

    return cosine


def encoder_hungarian_matching(args, feature_extractor, data1, data2, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                               left_image_encoded_features_last, right_image_encoded_features_last, use_path=False):
    if use_path:
        left_image = Image.open(data1).convert('RGB')
        right_image = Image.open(data2).convert('RGB')
    else:
        left_image = Image.fromarray(np.uint8(K.tensor_to_image(data1))).convert('RGB')
        right_image = Image.fromarray(np.uint8(K.tensor_to_image(data2))).convert('RGB')

    # Select boxes with the most confidence
    left_predicted_bboxes = np.asarray(left_predicted_bboxes)
    left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 4] > args.det_thres)[0]]
    right_predicted_bboxes = np.asarray(right_predicted_bboxes)
    right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 4] > args.det_thres)[0]]
    left_predicted_bboxes = left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
    right_predicted_bboxes = right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
    left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 2] > left_predicted_bboxes[:, 0])[0]]
    left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 3] > left_predicted_bboxes[:, 1])[0]]
    right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 2] > right_predicted_bboxes[:, 0])[0]]
    right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 3] > right_predicted_bboxes[:, 1])[0]]

    if len(left_predicted_bboxes) == 0 or len(right_predicted_bboxes) == 0:
        left_predicted_bboxes = torch.from_numpy(left_predicted_bboxes)
        right_predicted_bboxes = torch.from_numpy(right_predicted_bboxes)
        left_predicted_classes = torch.zeros(left_predicted_bboxes.shape[0])
        right_predicted_classes = torch.zeros(right_predicted_bboxes.shape[0])

        return ([[left_predicted_bboxes.cpu(), left_predicted_classes.cpu()]],
        [[right_predicted_bboxes.cpu(), right_predicted_classes.cpu()]],
        [left_target_bboxes.cpu()],
        [right_target_bboxes.cpu()])

    if args.cost_matrix_method == 'unet_encoder':
        pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        left_predicted_features = torch.cat([torch.flatten(pooling(feature_extractor.unet_model.encoder(cyws_transform(left_image.crop(bbox[:4])).unsqueeze(0).to(device))[-1]), 1) for bbox in left_predicted_bboxes], dim=0)
        right_predicted_features = torch.cat([torch.flatten(pooling(feature_extractor.unet_model.encoder(cyws_transform(right_image.crop(bbox[:4])).unsqueeze(0).to(device))[-1]), 1) for bbox in right_predicted_bboxes], dim=0)
    else:
        left_image_encoded_features = left_image_encoded_features_last[0]
        right_image_encoded_features = right_image_encoded_features_last[0]                                  
        left_image_encoded_features = torch.flatten(left_image_encoded_features.permute(0, 2, 3, 1), start_dim=1, end_dim=2)
        right_image_encoded_features = torch.flatten(right_image_encoded_features.permute(0, 2, 3, 1), start_dim=1, end_dim=2)

        left_predict_index = torch.floor(torch.from_numpy(np.asarray(left_predicted_bboxes))/32).to(device)
        right_predict_index = torch.floor(torch.from_numpy(np.asarray(right_predicted_bboxes))/32).to(device)

        left_predicted_features = []
        for idx in range(len(left_predict_index)):
            x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), 7), min(max(int(left_predict_index[idx][2]), 0), 7) + 1)
            y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), 7), min(max(int(left_predict_index[idx][3]), 0), 7) + 1)
            coor = torch.cartesian_prod(x_coor, y_coor)
            flatten_coor = torch.tensor(coor[:, 1]*8 + coor[:, 0], dtype=torch.long)
            left_predicted_features.append(torch.mean(left_image_encoded_features[:, flatten_coor, :], dim=1))
        left_predicted_features = torch.cat(left_predicted_features, dim=0)

        right_predicted_features = []
        for idx in range(len(right_predict_index)):
            x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), 7), min(max(int(right_predict_index[idx][2]), 0), 7) + 1)
            y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), 7), min(max(int(right_predict_index[idx][3]), 0), 7) + 1)
            coor = torch.cartesian_prod(x_coor, y_coor)
            flatten_coor = torch.tensor(coor[:, 0]*8 + coor[:, 1], dtype=torch.long)
            right_predicted_features.append(torch.mean(right_image_encoded_features[:, flatten_coor, :], dim=1))
        right_predicted_features = torch.cat(right_predicted_features, dim=0)

    similarity_map = 1.0 - cosine_similarity(left_predicted_features, right_predicted_features).detach().cpu().numpy()
    indices = linear_sum_assignment(similarity_map)

    left_predicted_hungarian = torch.cat([torch.from_numpy(np.append(left_predicted_bboxes[idx], idx)).unsqueeze(0) for idx in indices[0]], dim=0)
    right_predicted_hungarian = torch.cat([torch.from_numpy(np.append(right_predicted_bboxes[idx], idx)).unsqueeze(0) for idx in indices[1]], dim=0)
    left_predicted_classes = torch.zeros(len(indices[0]))
    right_predicted_classes = torch.zeros(len(indices[0]))

    return ([[left_predicted_hungarian.cpu(), left_predicted_classes.cpu()]],
        [[right_predicted_hungarian.cpu(), right_predicted_classes.cpu()]],
        [left_target_bboxes.cpu()],
        [right_target_bboxes.cpu()])


# Method combine both remove noise, matching pairs of boxes with very high iou, and hungarian algorithm
def matching_boxes(args, cyws_3d_model, depth_predictor, correspondence_extractor, feature_extractor,
                   data1, data2, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, 
                   left_image_encoded_features_last, right_image_encoded_features_last,
                   subplots, name, use_path=False):
    # Save original predict boxes

    # First step, we need to remove the noise
    if use_path:
        batch = create_batch_from_image_path([(data1, data2)])
    else:
        batch = create_batch_from_image_data([(data1, data2)])
    batch = fill_in_the_missing_information(batch, depth_predictor, correspondence_extractor)
    batch = prepare_batch_for_model(batch)
    M_1_to_2, M_2_to_1, transform_points_1_to_2, transform_points_2_to_1 = cyws_3d_model.registeration_module.transform_matrix(batch)
    if torch.all(torch.eq(M_1_to_2[0], torch.eye(3).to(device))):
        if args.use_encoder_feature:
            predict = encoder_hungarian_matching(args, feature_extractor, data1, data2, left_predicted_bboxes, right_predicted_bboxes, 
                                                 left_target_bboxes, right_target_bboxes, 
                                                 left_image_encoded_features_last, right_image_encoded_features_last, use_path=False)
        else:
            predict = emd_hungarian_matching(args, feature_extractor, data1, data2, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, use_path=False)

        if args.visualize:
            visualization(args, predict[0][0][0], predict[1][0][0], predict[2][0], predict[3][0], subplots, name)

        return predict

    if args.remove_noise:
        # Tranform from left to right
        points1 = []
        for bbox in left_predicted_bboxes:
            if bbox[4] >= args.det_thres:
                points1.append([bbox[0], bbox[1]])
                points1.append([bbox[2], bbox[3]])
        points1 = torch.from_numpy(np.asarray(points1)).to(device).reshape(-1, 2)/256.0
        points_1_to_2 = transform_points_1_to_2(points1, 0).reshape(-1, 4)*256.0
        points_1_to_2 = points_1_to_2.cpu()
        # Fix the situation where the transformation matrix estimated wrong
        if (points_1_to_2[:, 2:] >= points_1_to_2[:, :2]).all():
            # Remove the noise boxes in left and right images by setting detection threshold
            left_predicted_bboxes = np.asarray(left_predicted_bboxes) 
            left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 4] >= args.det_thres)[0]]
            left_predicted_bboxes = torch.from_numpy(left_predicted_bboxes)
            right_predicted_bboxes = np.asarray(right_predicted_bboxes)
            right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 4] >= args.det_thres)[0]]
            right_predicted_bboxes = torch.from_numpy(right_predicted_bboxes)
            # Save image after setting detection threshold
            # if args.visualize:
                # visualization(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes.cpu(), right_target_bboxes.cpu(), subplots, name.split('.')[0] + '_after_set_det.png')

            # Start calculate iou to remove noise boxes
            iou, union = box_iou(points_1_to_2, right_predicted_bboxes[:,:4])
            iou_matching = iou > args.iou_threshold
            iou_matching = iou_matching.long()
            left_boxes_idxs = torch.where(torch.sum(iou_matching, dim=1) > 0)[0]   # Find all noise boxes in the left image
            right_boxes_idxs = torch.where(torch.sum(iou_matching, dim=0) > 0)[0]   # Find all noise boxes in the right image
            left_predicted_bboxes = left_predicted_bboxes[left_boxes_idxs].numpy()
            right_predicted_bboxes = right_predicted_bboxes[right_boxes_idxs].numpy()
            left_predicted_bboxes = left_predicted_bboxes[left_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
            right_predicted_bboxes = right_predicted_bboxes[right_predicted_bboxes[:, 4].argsort()][::-1][:args.topk, :]
            left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 2] > left_predicted_bboxes[:, 0])[0]]
            left_predicted_bboxes = left_predicted_bboxes[np.where(left_predicted_bboxes[:, 3] > left_predicted_bboxes[:, 1])[0]]
            right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 2] > right_predicted_bboxes[:, 0])[0]]
            right_predicted_bboxes = right_predicted_bboxes[np.where(right_predicted_bboxes[:, 3] > right_predicted_bboxes[:, 1])[0]]
            left_predicted_bboxes = torch.from_numpy(left_predicted_bboxes.copy())
            right_predicted_bboxes = torch.from_numpy(right_predicted_bboxes.copy())

            if len(left_predicted_bboxes) == 0 or len(right_predicted_bboxes) == 0:
                left_predicted_classes = torch.zeros(left_predicted_bboxes.shape[0])
                right_predicted_classes = torch.zeros(right_predicted_bboxes.shape[0])

                if args.visualize:
                    if len(left_predicted_bboxes) == 0 and len(right_predicted_bboxes) > 0:
                        visualization(args, left_predicted_bboxes, left_predicted_bboxes, left_target_bboxes.cpu(), right_target_bboxes.cpu(), subplots, name)
                    elif len(left_predicted_bboxes) > 0 and len(right_predicted_bboxes) == 0:
                        visualization(args, right_predicted_bboxes, right_predicted_bboxes, left_target_bboxes.cpu(), right_target_bboxes.cpu(), subplots, name)
                    else:
                        visualization(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes.cpu(), right_target_bboxes.cpu(), subplots, name)
            
                return ([[left_predicted_bboxes.cpu(), left_predicted_classes.cpu()]],
                        [[right_predicted_bboxes.cpu(), right_predicted_classes.cpu()]],
                        [left_target_bboxes.cpu()],
                        [right_target_bboxes.cpu()])

            # Calculate for select pair of matching boxes which have iou larger than iou_match_threshold
            points1 = []
            for bbox in left_predicted_bboxes:
                points1.append([bbox[0], bbox[1]])
                points1.append([bbox[2], bbox[3]])
            points1 = torch.from_numpy(np.asarray(points1)).to(device).reshape(-1, 2)/256.0
            points_1_to_2 = transform_points_1_to_2(points1, 0).reshape(-1, 4)*256.0
            points_1_to_2 = points_1_to_2.cpu()

            left_predicted_bboxes_tmp = []
            right_predicted_bboxes_tmp = []
            left_remove_index = []
            right_remove_index = []
            iou, union = box_iou(points_1_to_2, right_predicted_bboxes[:,:4])
            iou_matching_mask = iou > args.iou_match_threshold
            iou_matching_mask = iou_matching_mask.long()
            # Find maximum matching in each row
            iou_max_row = torch.argmax(iou_matching_mask, dim=1)
            iou_max_col = torch.argmax(iou_matching_mask, dim=0)
            for i, idx in enumerate(iou_max_row):
                if iou_matching_mask[i, idx] == False:
                    continue
                else:
                    if iou_max_col[idx] == i:
                        left_predicted_bboxes_tmp.append(left_predicted_bboxes[i].detach().cpu().unsqueeze(0))
                        right_predicted_bboxes_tmp.append(right_predicted_bboxes[idx].detach().cpu().unsqueeze(0))
                        left_remove_index.append(i)
                        right_remove_index.append(idx)

            left_keep_index = [i for i in range(len(left_predicted_bboxes)) if i not in left_remove_index]
            right_keep_index = [i for i in range(len(right_predicted_bboxes)) if i not in right_remove_index]
            if len(left_keep_index) > 0  and len(right_keep_index) > 0:
                # Hungarian matching for remaining boxes
                if use_path:
                    left_image = Image.open(data1).convert('RGB')
                    right_image = Image.open(data2).convert('RGB')
                else:
                    left_image = Image.fromarray(np.uint8(K.tensor_to_image(data1))).convert('RGB')
                    right_image = Image.fromarray(np.uint8(K.tensor_to_image(data2))).convert('RGB')
                if args.use_encoder_feature:
                    if args.cost_matrix_method == 'unet_encoder':
                        pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
                        left_predicted_features = torch.cat([torch.flatten(pooling(feature_extractor.unet_model.encoder(cyws_transform(left_image.crop(bbox[:4].numpy())).unsqueeze(0).to(device))[-1]), 1) for bbox in left_predicted_bboxes[left_keep_index]], dim=0)
                        right_predicted_features = torch.cat([torch.flatten(pooling(feature_extractor.unet_model.encoder(cyws_transform(right_image.crop(bbox[:4].numpy())).unsqueeze(0).to(device))[-1]), 1) for bbox in right_predicted_bboxes[right_keep_index]], dim=0)
                    else:
                        left_image_encoded_features = left_image_encoded_features_last[0]  
                        right_image_encoded_features = right_image_encoded_features_last[0]                               
                        left_image_encoded_features = torch.flatten(left_image_encoded_features.permute(0, 2, 3, 1), start_dim=1, end_dim=2)
                        right_image_encoded_features = torch.flatten(right_image_encoded_features.permute(0, 2, 3, 1), start_dim=1, end_dim=2)

                        left_predict_index = torch.floor(torch.from_numpy(np.asarray(left_predicted_bboxes[left_keep_index]))/32).to(device)
                        right_predict_index = torch.floor(torch.from_numpy(np.asarray(right_predicted_bboxes[right_keep_index]))/32).to(device)

                        left_predicted_features = []
                        for idx in range(len(left_predict_index)):
                            x_coor = torch.arange(min(max(int(left_predict_index[idx][0]), 0), 7), min(max(int(left_predict_index[idx][2]), 0), 7) + 1)
                            y_coor = torch.arange(min(max(int(left_predict_index[idx][1]), 0), 7), min(max(int(left_predict_index[idx][3]), 0), 7) + 1)
                            coor = torch.cartesian_prod(x_coor, y_coor)
                            flatten_coor = torch.tensor(coor[:, 1]*8 + coor[:, 0], dtype=torch.long)
                            left_predicted_features.append(torch.mean(left_image_encoded_features[:, flatten_coor, :], dim=1))
                        left_predicted_features = torch.cat(left_predicted_features, dim=0)

                        right_predicted_features = []
                        for idx in range(len(right_predict_index)):
                            x_coor = torch.arange(min(max(int(right_predict_index[idx][0]), 0), 7), min(max(int(right_predict_index[idx][2]), 0), 7) + 1)
                            y_coor = torch.arange(min(max(int(right_predict_index[idx][1]), 0), 7), min(max(int(right_predict_index[idx][3]), 0), 7) + 1)
                            coor = torch.cartesian_prod(x_coor, y_coor)
                            flatten_coor = torch.tensor(coor[:, 0]*8 + coor[:, 1], dtype=torch.long)
                            right_predicted_features.append(torch.mean(right_image_encoded_features[:, flatten_coor, :], dim=1))
                        right_predicted_features = torch.cat(right_predicted_features, dim=0)

                    similarity_map = 1.0 - cosine_similarity(left_predicted_features, right_predicted_features).detach().cpu().numpy()
                    indices = linear_sum_assignment(similarity_map)             
                else:
                    left_crop_bboxes = torch.cat([emd_transform(left_image.crop(bbox[:4]*args.ratio)).unsqueeze(0) for bbox in left_predicted_bboxes[left_keep_index].detach().cpu().numpy()], dim=0)
                    right_crop_bboxes = torch.cat([emd_transform(right_image.crop(bbox[:4]*args.ratio)).unsqueeze(0) for bbox in right_predicted_bboxes[right_keep_index].detach().cpu().numpy()], dim=0)
                    data = torch.cat((left_crop_bboxes, right_crop_bboxes)).to(device)
                    feature_extractor.mode = 'encoder'
                    data = feature_extractor(data)
                    k = left_crop_bboxes.shape[0]
                    data_shot, data_query = data[:k], data[k:]
                    similarity_map = 1.0 - feature_extractor.get_similiarity_map(data_shot, data_query)
                    similarity_map = torch.sum(similarity_map, dim=[2, 3]).detach().cpu().numpy()
                    indices = linear_sum_assignment(similarity_map)
                    indices = [indices[1], indices[0]]

                for i, idx in enumerate(indices[0]):
                    left_predicted_bboxes_tmp.append(left_predicted_bboxes[left_keep_index][idx].detach().cpu().unsqueeze(0))
                for i, idx in enumerate(indices[1]):
                    right_predicted_bboxes_tmp.append(right_predicted_bboxes[right_keep_index][idx].detach().cpu().unsqueeze(0))

            left_predicted_emd_hungarian = torch.cat(left_predicted_bboxes_tmp, dim=0)
            right_predicted_emd_hungarian = torch.cat(right_predicted_bboxes_tmp, dim=0)
            left_predicted_classes = torch.zeros(left_predicted_emd_hungarian.shape[0])
            right_predicted_classes = torch.zeros(left_predicted_emd_hungarian.shape[0])
            
            if args.visualize:
                visualization(args, left_predicted_emd_hungarian.cpu(), right_predicted_emd_hungarian.cpu(), left_target_bboxes.cpu(), right_target_bboxes.cpu(), subplots, name)

            return ([[left_predicted_emd_hungarian.cpu(), left_predicted_classes.cpu()]],
                    [[right_predicted_emd_hungarian.cpu(), right_predicted_classes.cpu()]],
                    [left_target_bboxes.cpu()],
                    [right_target_bboxes.cpu()])
        else:
            if args.use_encoder_feature:
                predict = encoder_hungarian_matching(args, feature_extractor, data1, data2, left_predicted_bboxes, right_predicted_bboxes, 
                                                 left_target_bboxes, right_target_bboxes, 
                                                 left_image_encoded_features_last, right_image_encoded_features_last, use_path=False)
            else:
                predict = emd_hungarian_matching(args, feature_extractor, data1, data2, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, use_path=False)

            if args.visualize:
                visualization(args, predict[0][0][0], predict[1][0][0], predict[2][0], predict[3][0], subplots, name)

            return predict
    else:
        if args.use_encoder_feature:
            predict = encoder_hungarian_matching(args, feature_extractor, data1, data2, left_predicted_bboxes, right_predicted_bboxes, 
                                                 left_target_bboxes, right_target_bboxes, 
                                                 left_image_encoded_features_last, right_image_encoded_features_last, use_path=False)
        else:
            predict = emd_hungarian_matching(args, feature_extractor, data1, data2, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, use_path=False)

        if args.visualize:
            visualization(args, predict[0][0][0], predict[1][0][0], predict[2][0], predict[3][0], subplots, name)
        
        return predict


def visualization(args, left_predicted_bboxes, right_predicted_bboxes, left_target_bboxes, right_target_bboxes, subplots, name):
    # Creating folder to save the results
    os.makedirs(args.dst, exist_ok=True)
    
    left_predicted_bboxes = left_predicted_bboxes.numpy().tolist()
    right_predicted_bboxes = right_predicted_bboxes.numpy().tolist()
    left_target_bboxes = left_target_bboxes.numpy()
    right_target_bboxes = right_target_bboxes.numpy()

    # Draw the ground-truth
    if args.change:
        for index, bboxes in zip([0, 1], [left_target_bboxes, right_target_bboxes]):
            if len(bboxes) == 0:
                continue
            for idx, bbox in enumerate(bboxes):
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=2, edgecolor=color_map['10'], facecolor="none", linestyle='dashed'
                )
                subplots[index].add_patch(rect)

    # Draw the bounding boxes for left and right images after post processing
    for index, bboxes in zip([0, 1], [left_predicted_bboxes, right_predicted_bboxes]):
        for idx, bbox in enumerate(bboxes):
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            rect = patches.Rectangle(
                bbox[:2], w, h, linewidth=2, edgecolor=color_map[str(idx)], facecolor="none"
            )
            subplots[index].add_patch(rect)

    for subplot in subplots:
        subplot.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(args.dst, name.split('.')[0] + '.pdf'), format='pdf', bbox_inches='tight', pad_inches=0) 


def visualization_after_detection(args, left_predicted_bboxes, right_predicted_bboxes, 
                  left_target_bboxes, right_target_bboxes, subplots, name):
    os.makedirs(args.dst, exist_ok=True)

    # Draw the ground-truth
    if args.change:
        for index, bboxes in zip([0, 1], [left_target_bboxes, right_target_bboxes]):
            for bbox in bboxes:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    bbox[:2], w, h, linewidth=2, edgecolor=color_map['10'], facecolor="none", linestyle='dashed'
                )
                subplots[index].add_patch(rect)

    # Draw the results
    for index, bboxes in zip([0, 1], [left_predicted_bboxes, right_predicted_bboxes]):
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if bbox[4] < args.det_thres:
                continue
            rect = patches.Rectangle(
                bbox[:2], w, h, linewidth=2, edgecolor=color_map['2'], facecolor="none"
            )
            subplots[index].add_patch(rect)

    for subplot in subplots:
        subplot.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(args.dst, name.split('.')[0] + '.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)