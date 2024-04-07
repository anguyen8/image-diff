import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from loguru import logger as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from data.datamodule import DataModule
from models.centernet_with_coam import CenterNetWithCoAttention
from models.coatss_with_coam import CoATSSWithCoAttention
from utils.general import get_easy_dict_from_yaml_file

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
 
warnings.filterwarnings("ignore")


@rank_zero_only
def print_args(configs):
    L.log("INFO", configs)


def train(configs, model, logger, datamodule, callbacks=None):
    L.log("INFO", f"Training model.")
    trainer = pl.Trainer.from_argparse_args(
        configs,
        logger=logger,
        strategy=DDPPlugin(find_unused_parameters=False),
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        benchmark=False,
        profiler="simple",
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer, trainer.checkpoint_callback.best_model_path


def test(configs, model, logger, datamodule, checkpoint_path, callbacks=None):
    L.log("INFO", f"Testing model.")
    tester = pl.Trainer.from_argparse_args(
        configs, logger=logger, callbacks=callbacks, benchmark=True
    )
    tester.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

def get_logging_callback_manager(args):
    if args.method == "centernet":
        from models.centernet_with_coam import WandbCallbackManager

        return WandbCallbackManager(args)
    elif args.method == 'coatss':
        from models.centernet_with_coam import WandbCallbackManager

        return WandbCallbackManager(args)

    raise NotImplementedError(f"Given method ({args.method}) not implemented!")


if __name__ == "__main__":
    os.makedirs('./logs', exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="centernet")
    parser.add_argument("--no_logging", action="store_true", default=True)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--test_from_checkpoint", type=str, default="")
    parser.add_argument("--quick_prototype", action="store_true", default=True)
    parser.add_argument("--load_weights_from", type=str, default=None)
    parser.add_argument("--config_file", type=str, default='/home/hung/difference_localization/configs/detection_resnet50_3x_coam_layers_affine.yml')
    parser.add_argument("--experiment_name", type=str, default='Centernet_from_scratch_with_no_change_training')
    parser.add_argument("--iou_threshold", help="Threshold for remove boxes from left and right images", default=0.0)
    parser.add_argument("--iou_match_threshold", help="Threshold for accept two boxes in left and right images are corresponding", default=0.1)
    parser.add_argument("--iou_match_target_threshold", help="Threshold for a predicted is classified with target boxes", default=0.5)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--lamda_l1", default=3.0)
    parser.add_argument("--lamda_giou", default=3.0)
    parser.add_argument("--lamda_matching", default=2.0)
    parser.add_argument("--use_ground_truth", default=None)
    parser.add_argument("--use_coattention_feature", default=False)
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
    parser.add_argument('-emd_model_dir', type=str, default='/home/hung/difference_localization/checkpoints/deepemd_trained_model/miniimagenet/fcn/max_acc.pth')
    args, _ = parser.parse_known_args()
    if args.method == "centernet":
        parser = CenterNetWithCoAttention.add_model_specific_args(parser)
    elif args.method == 'coatss':
        parser = CoATSSWithCoAttention.add_model_specific_args(parser)
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

    if configs.quick_prototype:
        configs.limit_train_batches = 2
        configs.limit_val_batches = 2
        configs.limit_test_batches = 2
        configs.max_epochs = 1

    print_args(configs)

    pl.seed_everything(1, workers=True)
    datamodule = DataModule(configs)
    if configs.method == "centernet":
        model = CenterNetWithCoAttention(configs)
    elif configs.method == 'coatss':
        model = CoATSSWithCoAttention(configs)


    logger = None
    callbacks = [get_logging_callback_manager(configs)]
    if not configs.no_logging:
        logger = WandbLogger(
            project="Change_Detection_ResNet_50_Train_with_200_epochs",
            id=configs.wandb_id,
            save_dir="./logs",
            name=configs.experiment_name,
            log_model='all',
        )
        callbacks.append(ModelCheckpoint(monitor="val/overall_loss", mode="min", save_last=True))

    trainer = None
    if configs.test_from_checkpoint == "":
        # train the model and store the path to the best model (as per the validation set)
        # Note: multi-GPU training is supported.
        trainer, test_checkpoint_path = train(configs, model, logger, datamodule, callbacks)
        # test the best model exactly once on a single GPU
        torch.distributed.destroy_process_group()
    else:
        # test the given model checkpoint
        test_checkpoint_path = configs.test_from_checkpoint

    configs.gpus = 1
    if trainer is None or trainer.global_rank == 0:
        test(
            configs,
            model,
            logger,
            datamodule,
            test_checkpoint_path if test_checkpoint_path != "" else None,
            callbacks,
        )

    
