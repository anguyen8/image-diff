import numpy as np
import torch
from collections import defaultdict
from .voc_eval import BoxList, boxlist_iou
from tqdm import tqdm


# Some detail of predict_boxes and ground_truth_boxes
# predict_boxes: list of batch of predict boxes: PxNx4
# ground_truth_boxes: list of batch ground truth boxes: PxMx4
# P: Number of batches
# N: Number of predict boxes
# M: Number of ground truth boxes
# We store the the predict boxes for the left images, and the right images for each pairs of images
# Similar for the ground truth

def count_matching_boxes(multiple_test_set_outputs, iou_thresh=0.5):
    # iterate over all the batches for the current test set
    for test_set_batch_outputs in multiple_test_set_outputs:
        predicted_bboxes = []
        target_bboxes = []
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
                    # if len(bboxes_per_image) == 0:
                    #     continue
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
                    bbox_list = BoxList(
                        bboxes_per_image,
                        image_size=(256, 256),
                        mode="xyxy",
                    )
                    bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                    bbox_list.add_field("difficult", torch.zeros(bboxes_per_image.shape[0]))
                    target_bboxes.append(bbox_list)

        # Find the matching and ground truth indexes
        matches, gt_indexes, order_scores, predict_scores, num_gt_boxes = find_matching(target_bboxes, predicted_bboxes, iou_thresh=iou_thresh)

        # Check the number of pairs of images
        num_pair = int(len(matches)/2)

        # Cound the number of matching pairs
        matching_boxes = []
        for idx in tqdm(range(num_pair)):
            left_match = np.asarray(matches[idx*2][1])
            right_match = np.asarray(matches[idx*2+1][1])
            left_order_score = np.asarray(order_scores[idx*2][1])
            right_order_score = np.asarray(order_scores[idx*2+1][1])
            left_pred_score = np.asarray(predict_scores[idx*2][1])
            right_pred_score = np.asarray(predict_scores[idx*2+1][1])
            left_gt_index = np.asarray(gt_indexes[idx*2])
            right_gt_index = np.asarray(gt_indexes[idx*2+1])

            count = 0
            for idx in range(len(left_match)):
                if left_match[idx] == 1:
                    r_idx = np.where(right_gt_index == left_gt_index[idx])[0]
                    if len(r_idx) > 0:
                        left_score = left_order_score[idx]
                        right_score = right_order_score[r_idx[0]]
                        if np.where(left_pred_score == left_score)[0] == np.where(right_pred_score == right_score)[0]:
                            count += 1
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            matching_boxes.append(count)

        print('Num of matching boxes: {}'.format(sum(matching_boxes)))
        print('Num of ground truth boxes: {}'.format(sum(num_gt_boxes)/2))
        

def find_matching(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    """

    matches = []
    gt_indexes = []
    num_gt_boxes = []
    predict_scores = []
    order_scores = []
    count = 0
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        n_pos = defaultdict(list)
        predict_score = defaultdict(list)
        score = defaultdict(list)
        match = defaultdict(list)

        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()
        num_gt_boxes.append(len(gt_bbox))

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            predict_score[l].extend(pred_score_l)

            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()

            gt_index = iou.argmax(axis=1)

            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            gt_indexes.append(gt_index)
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

            matches.append(match)
            order_scores.append(score)
            predict_scores.append(predict_score)

        count += 1

    return matches, gt_indexes, order_scores, predict_scores, num_gt_boxes


if __name__ == '__main__':
    data = np.load('./matching_boxes_data/multiple_test_set_outputs.npy', allow_pickle=True)
    count_matching_boxes([data])
