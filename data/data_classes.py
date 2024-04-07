import os
from pathlib import Path
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset
import torch
from .augmentation import AugmentationPipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from scipy.ndimage import label as label_connected_components
from einops import rearrange
from torchvision.ops import masks_to_boxes
import h5py
import kornia as K
import shapely.geometry

import utils_cyws.geometry


class COCOPair(Dataset):
    def __init__(self, args, method, path_to_dataset, index, mode='identity'):
        self.args = args
        self.mode = mode
        self.path_to_dataset = path_to_dataset
        self.path_to_image1 = './demo_images/img1.png'
        self.path_to_image2 = './demo_images/img2.jpg'
        self.split = "test"
        self.index = index
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.image_augmentations = AugmentationPipeline(
            mode='test', path_to_dataset=path_to_dataset, image_transformation=mode
        )

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        w, h = pil_image.size
        return image_as_tensor, (w, h)

    def get_inpainted_objects_bitmap_from_image_path(self, image_path, bit_length):
        if "inpainted" not in image_path:
            return 0
        bitmap_string = image_path.split("mask")[1].split(".")[0]
        if bitmap_string == "":
            return (2**bit_length) - 1
        return int(bitmap_string)

    def get_dummy_target_annotations_in_coco_format(self):
        # replace this if GT annotations are known
        x, y, w, h = 0, 0, 5, 5
        four_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        coco_annotation = {
            "bbox": [*four_corners[0], *four_corners[2]],
            "segmentation": [four_corners.reshape(-1)],
        }
        return [coco_annotation]

    def __len__(self):
        return 1

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        image1, image1_size = self.read_image_as_tensor(self.path_to_image1)
        image2, image2_size = self.read_image_as_tensor(self.path_to_image2)

        annotations = np.load(os.path.join(self.path_to_dataset, "metadata", self.index + '.npy'), allow_pickle=True)
        image1_image_inpainted_objects = self.get_inpainted_objects_bitmap_from_image_path(
            os.path.join("images_and_masks", self.path_to_image1.split('/')[-1]), len(annotations)
        )
        if self.args.change:
            image2_image_inpainted_objects = self.get_inpainted_objects_bitmap_from_image_path(
                os.path.join("inpainted", self.path_to_image2.split('/')[-1]), len(annotations)
            )
        else:
            image2_image_inpainted_objects = self.get_inpainted_objects_bitmap_from_image_path(
            os.path.join("images_and_masks", self.path_to_image1.split('/')[-1]), len(annotations)
            )
        changed_objects = image1_image_inpainted_objects ^ image2_image_inpainted_objects
        change_objects_indices = np.array(
            [x == "1" for x in bin(changed_objects)[2:].zfill(len(annotations))]
        )

        if self.args.change:
            annotations = annotations[change_objects_indices]

        image1_image_as_tensor, image2_image_as_tensor, transformed_image1_target_annotations, transformed_image2_target_annotations \
            = self.image_augmentations(image1, image2, annotations, self.index)

        return {
            "image1": image1_image_as_tensor,
            "image2": image2_image_as_tensor,
            "image1_target_annotations": transformed_image1_target_annotations,
            "image2_target_annotations": transformed_image2_target_annotations,
            "image1_size": image1_size,
            "image2_size": image2_size,
            "image1_path": self.path_to_image1,
            "image2_path": self.path_to_image2
        }
    

class STDPair(Dataset):
    def __init__(self, args, method, path_to_dataset, mode='identity'):
        self.args = args
        self.mode = mode
        self.path_to_dataset = path_to_dataset
        self.path_to_image1 = './demo_images/img1.png'
        self.path_to_image2 = './demo_images/img2.jpg'
        self.image_id = None
        self.split = "test"
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.annotations = self.get_annotations()
        self.image_ids = list(self.annotations.keys())

    def get_annotations(self):
        return np.load(
            os.path.join(self.path_to_dataset, "annotations.npy"), allow_pickle=True
        ).item()

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def get_target_annotations_in_coco_format(self, bboxes):
        coco_annotations = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            four_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            coco_annotation = {
                "bbox": [*four_corners[0], *four_corners[2]],
                "segmentation": [four_corners.reshape(-1)],
            }
            coco_annotations.append(coco_annotation)
        return coco_annotations
    
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
        return 1

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        image1 = self.read_image_as_tensor(self.path_to_image1)
        if self.args.change:
            image2 = self.read_image_as_tensor(self.path_to_image2)
        else:
            image2 = self.read_image_as_tensor(self.path_to_image1)

        annotation = self.get_target_annotations_in_coco_format(self.annotations[self.image_id])
        if self.mode == "projective":
            image1, target_annotations_1 = self.random_perspective(
                image1, deepcopy(annotation), "image1", item_index
            )
            image2, target_annotations_2 = self.random_perspective(
                image2, deepcopy(annotation), "image2", item_index
            )
        else:
            target_annotations_1 = annotation
            target_annotations_2 = annotation

        return {
            "image1": image1,
            "image2": image2,
            "image1_path": self.path_to_image1,
            "image2_path": self.path_to_image2,
            "image1_target_annotations": target_annotations_1,
            "image2_target_annotations": target_annotations_2,
        }
    

class KubricPair(Dataset):
    def __init__(self, args, method, path_to_dataset, mode='identity'):
        self.args = args
        self.mode = mode
        self.path_to_dataset = path_to_dataset
        self.mask1_path = None
        self.mask2_path = None
        self.path_to_image1 = './demo_images/img1.png'
        self.path_to_image2 = './demo_images/img2.jpg'
        self.split = "test"
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.data = self.get_data_info(path_to_dataset)

    def get_data_info(self, path_to_dataset):
        if os.path.exists(os.path.join(path_to_dataset, "metadata.npy")):
            return np.load(os.path.join(path_to_dataset, "metadata.npy"), allow_pickle=True)
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
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

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
        return 1

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        image1_as_tensor = self.read_image_as_tensor(self.path_to_image1)
        if self.args.change:
            image2_as_tensor = self.read_image_as_tensor(self.path_to_image2)
        else:
            image2_as_tensor = self.read_image_as_tensor(self.path_to_image1)
        image1_target_annotation = self.get_target_annotations_in_coco_format(self.mask1_path)
        if self.args.change:
            image2_target_annotation = self.get_target_annotations_in_coco_format(self.mask2_path)
        else:
            image2_target_annotation = self.get_target_annotations_in_coco_format(self.mask1_path)

        if self.mode == "projective":
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
            "image1": image1_as_tensor,
            "image2": image2_as_tensor,
            "image1_path": self.path_to_image1,
            "image2_path": self.path_to_image2,
            "image1_target_annotations": target_annotations_1,
            "image2_target_annotations": target_annotations_2,
        }


class SynthtextPair(Dataset):
    def __init__(self, args, method, path_to_dataset, item_index, mode='identity'):
        self.args = args
        self.mode = mode
        self.item_index = item_index
        self.path_to_dataset = path_to_dataset
        self.split = "test"
        self.synthetic_images = h5py.File(os.path.join(path_to_dataset, "synthtext-change.h5"), "r")
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.original_image_names, self.synthetic_image_names = self.get_paths_of_test_images()

    def get_paths_of_test_images(self):
        h5_keys = sorted(self.synthetic_images["data"].keys())
        synthetic_image_names = []
        original_image_names = []
        bg_img_directory_path = os.path.join(self.path_to_dataset, "bg_img")
        files_in_bg_img_directory = os.listdir(bg_img_directory_path)
        for key in h5_keys:
            synthetic_image_names.append(key)
            synth_image_name_and_extension = key.split(".")
            original_image_name_with_extension = [
                filename
                for filename in files_in_bg_img_directory
                if filename.split(".")[0] == synth_image_name_and_extension[0]
            ][0]
            original_image_names.append(original_image_name_with_extension)
        return original_image_names, synthetic_image_names

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def get_target_annotations_in_coco_format(self, bboxes):
        """
        bboxes.shape: 2x4xN
            2 -> x,y
            4 -> four corners (clockwise, starting from top left)
            N -> number of boxes
        """
        bboxes = np.array(bboxes)
        if len(bboxes.shape) == 2:
            bboxes = bboxes[..., np.newaxis]
        bboxes = rearrange(bboxes, "two four n -> n four two")
        annotations = []
        for bbox in bboxes:
            x, y = bbox[:, 0], bbox[:, 1]
            annotation = {
                "bbox": [min(x), min(y), max(x), max(y)],
                "segmentation": [bbox.reshape(-1)],
            }
            annotations.append(annotation)
        return annotations
    
    def random_perspective(self, image_as_tensor, annotations, type_of_image, image_index):
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
        return 1

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    def __base_getitem__(self, item_index):
        original_image_name = self.original_image_names[self.item_index]
        original_image_path = os.path.join(self.path_to_dataset, f"bg_img/{original_image_name}")
        original_image_as_tensor = self.read_image_as_tensor(original_image_path)
        synth_image_name = self.synthetic_image_names[self.item_index]
        synth_image_as_tensor = (
            K.image_to_tensor(self.synthetic_images["data"][synth_image_name][...])
            .squeeze()
            .float()
            / 255.0
        )
        original_image_as_tensor = K.geometry.transform.resize(
            original_image_as_tensor, synth_image_as_tensor.shape[-2:]
        )
        if not self.args.change:
            synth_image_as_tensor = original_image_as_tensor

        change_bboxes = self.synthetic_images["data"][synth_image_name].attrs["wordBB"]
        target_annotations = self.get_target_annotations_in_coco_format(change_bboxes)
        if self.args.image_transformation == "projective":
            original_image_as_tensor, target_annotations_1 = self.random_perspective(
                original_image_as_tensor, deepcopy(target_annotations), "original", item_index
            )
            synth_image_as_tensor, target_annotations_2 = self.random_perspective(
                synth_image_as_tensor, deepcopy(target_annotations), "synth", item_index
            )
        else:
            target_annotations_1 = target_annotations
            target_annotations_2 = target_annotations

        return {
            "image1": original_image_as_tensor.squeeze(),
            "image2": synth_image_as_tensor.squeeze(),
            "image1_path": os.path.join(self.path_to_dataset, f"bg_img/{original_image_name}"),
            "image2_path": None,
            "image1_target_annotations": target_annotations_1,
            "image2_target_annotations": target_annotations_2,
        }