from __future__ import annotations

import os
from pathlib import Path

import torch
import kornia as K
import numpy as np
import shapely.geometry
from einops import rearrange
from PIL import Image
from scipy.ndimage import label as label_connected_components
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import pil_to_tensor

import utils_cyws.geometry


class KubricChange(Dataset):
    def __init__(self, path_to_dataset, split, method, change, image_transformation='projective'):
        assert split == "test"
        self.split = split
        self.change = change
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.data = self.get_data_info(path_to_dataset)
        self.image_transformation = image_transformation
        self.path_to_dataset = path_to_dataset

    def get_data_info(self, path_to_dataset):
        # if os.path.exists(os.path.join(path_to_dataset, "metadata.npy")):
        #     return np.load(os.path.join(path_to_dataset, "metadata.npy"), allow_pickle=True)
        image_1, image_2, mask_1, mask_2 = [], [], [], []
        for file in os.listdir(path_to_dataset):
            file_without_extension = file.split(".")[0]
            id = file_without_extension.split("_")[-1]
            if id == "00000":
                mask_1.append(os.path.join(path_to_dataset, file))
            elif id == "00001":
                mask_2.append(os.path.join(path_to_dataset, file))
            elif id == "0":
                image_1.append(os.path.join(path_to_dataset, file))
            elif id == "1":
                image_2.append(os.path.join(path_to_dataset, file))
            else:
                continue
        assert len(image_1) == len(image_2) == len(mask_1) == len(mask_2)
        image_1, image_2, mask_1, mask_2 = (
            sorted(image_1),
            sorted(image_2),
            sorted(mask_1),
            sorted(mask_2),
        )
        data = np.array(list(zip(image_1, image_2, mask_1, mask_2)))
        np.save(os.path.join(path_to_dataset, "metadata"), data)
        return data

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor
    
    def random_perspective(self, image_as_tensor, annotations, type_of_image, image_index):
        # aug = K.augmentation.RandomPerspective(p=1.0, return_transform=True)
        aug = K.augmentation.RandomPerspective(p=1.0)
        precomputed_augmentation_path = os.path.join(
            self.path_to_dataset, f"projective_augmentations/{type_of_image}/{image_index}.params"
        )
        image_as_tensor = rearrange(image_as_tensor, "... -> 1 ...")
        if os.path.exists(precomputed_augmentation_path):
            augmentation_params = torch.load(precomputed_augmentation_path)
        else:
            aug_params = aug.generate_parameters(image_as_tensor.shape)
            augmentation_params = {"projective": aug_params}
            Path(precomputed_augmentation_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(augmentation_params, precomputed_augmentation_path)
        image_as_tensor = aug(
            image_as_tensor, params=augmentation_params["projective"]
        )
        transformation = aug.transform_matrix
        for annotation in annotations:
            bbox = rearrange(torch.Tensor(annotation["bbox"]), "four -> 1 four")
            bbox = K.geometry.bbox.transform_bbox(transformation, bbox)[0]
            annotation["bbox"] = bbox
            annotation[
                "segmentation"
            ] = utils_cyws.geometry.convert_shapely_polygon_into_coco_segmentation(
                shapely.geometry.box(*bbox)
            )
        return image_as_tensor.squeeze(), annotations

    def __len__(self):
        """
        Returns the number of testing images.
        """
        return len(self.data)

    def get_target_annotations_in_coco_format(self, mask_path):
        pil_image = Image.open(mask_path)
        mask_as_np_array = np.array(pil_image)
        (
            connected_components,
            number_of_components,
        ) = label_connected_components(mask_as_np_array)
        masks = []
        for i in range(number_of_components):
            masks.append(connected_components == i + 1)
        masks = rearrange(masks, "c h w -> h w c")
        masks_as_tensor = K.image_to_tensor(masks)
        bboxes = masks_to_boxes(masks_as_tensor)
        coco_annotations = []
        for bbox in bboxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            four_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            coco_annotation = {
                "bbox": [*four_corners[0], *four_corners[2]],
                "segmentation": [four_corners.reshape(-1)],
            }
            coco_annotations.append(coco_annotation)
        return coco_annotations

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        """
        Returns 3 things:
        a reference image (3xRxC),
        a query image (3xRxC), and
        a list of coco-format change annotations corresponding to the query image.
        """
        image1_path, image2_path, mask1_path, mask2_path = self.data[item_index]
        image1_as_tensor = self.read_image_as_tensor(image1_path)
        if self.change:
            image2_as_tensor = self.read_image_as_tensor(image2_path)
        else:
            image2_as_tensor = self.read_image_as_tensor(image1_path)
        image1_target_annotation = self.get_target_annotations_in_coco_format(mask1_path)
        if self.change:
            image2_target_annotation = self.get_target_annotations_in_coco_format(mask2_path)
        else:
            image2_target_annotation = self.get_target_annotations_in_coco_format(mask1_path)

        if self.image_transformation == "projective":
            image1_as_tensor, target_annotations_1 = self.random_perspective(
                image1_as_tensor, image1_target_annotation, "image1", item_index
            )
            image2_as_tensor, target_annotations_2 = self.random_perspective(
                image2_as_tensor, image2_target_annotation, "image2", item_index
            )
        else:
            target_annotations_1 = image1_target_annotation
            target_annotations_2 = image2_target_annotation

        return {
            "image1": image1_as_tensor.squeeze(),
            "image2": image2_as_tensor.squeeze(),
            "image1_target_annotations": target_annotations_1,
            "image2_target_annotations": target_annotations_2,
            "image1_path": None,
            "image2_path": None,
        }
