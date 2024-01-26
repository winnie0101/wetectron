# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import warnings
import torch
import wandb
from config import cfg
from data.build import make_data_loader
from solver import make_lr_scheduler, make_lr_cdb_scheduler
from solver import make_optimizer, make_cdb_optimizer
from tools.engine.inference import inference
from tools.engine.trainer import do_train, do_train_cdb
from models.generalized_rcnn import GeneralizedRCNN
from utils.checkpoint import DetectronCheckpointer
from utils.collect_env import collect_env_info
from utils.distributed import synchronize, get_rank, is_main_process
from utils.logger import setup_logger
from utils.miscellaneous import mkdir, save_config, seed_all_rng
from utils.metric_logger import (MetricLogger, WandbLogger)
from models.cdb import ConvConcreteDB


def setup_wandb(cfg):
    if is_main_process():
        wandb.init(project="wetectron", 
                   config=cfg
                )
        return wandb
    else:
        return None

def train(cfg, local_rank, distributed, use_wandb=False):
    model = GeneralizedRCNN(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )

    arguments = {"iteration": 0, "iter_size": cfg.SOLVER.ITER_SIZE}
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    if use_wandb:
        wandb = setup_wandb(cfg)
        if wandb is not None:
            print("wandb is not None")
            wandb_logger = WandbLogger(wandb, start_iter=arguments['iteration'], delimiter="  ")
            meters = wandb_logger
        else:
            meters = MetricLogger(delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")
        
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        meters
    )

    if use_wandb and wandb is not None:
        wandb.finish()
    return model


def train_cdb(cfg, local_rank, distributed, use_tensorboard=False, use_wandb=False):
    device = torch.device(cfg.MODEL.DEVICE)

    model = GeneralizedRCNN(cfg)
    model.to(device)

    model_cdb = ConvConcreteDB(cfg, model.backbone.out_channels)
    model_cdb.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    optimizer_cdb = make_cdb_optimizer(cfg, model_cdb)
    scheduler_cdb = make_lr_cdb_scheduler(cfg, optimizer_cdb)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )
        model_cdb = torch.nn.parallel.DistributedDataParallel(
            model_cdb, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False, find_unused_parameters=True,
        )

    arguments = {"iteration": 0}
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    print("checkpoint_period: ", checkpoint_period)

    # TODO: check whether the *_cdb is properly loaded for inference when using 1 GPU
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, model_cdb=model_cdb
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    
    if use_wandb:
        wandb = setup_wandb(cfg)
        if wandb is not None:
            wandb_logger = WandbLogger(wandb, start_iter=arguments['iteration'], delimiter="  ")
            meters = wandb_logger
        else:
            print("wandb is None")
            meters = MetricLogger(delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")

    do_train_cdb(
        model, model_cdb,
        data_loader,
        optimizer, optimizer_cdb,
        scheduler, scheduler_cdb,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        meters,
        cfg
    )

    if use_wandb and wandb is not None:
        wandb.finish()
    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="wetectron training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use-wandb",
        dest="use_wandb",
        help="Use weight&bias logger (Requires wandb installed)",
        action="store_true",
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    update_iters()
    cfg.freeze()

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + get_rank())

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("wetectron", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    if cfg.DB.METHOD == "concrete":
        model = train_cdb(
            cfg=cfg,
            local_rank=args.local_rank,
            distributed=args.distributed,
            use_wandb=args.use_wandb
        )
    else:
        model = train(
            cfg=cfg,
            local_rank=args.local_rank,
            distributed=args.distributed,
            use_wandb=args.use_wandb
        )

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


def update_iters():
    if cfg.SOLVER.ITER_SIZE > 1:
        assert cfg.DB.METHOD != "concrete", "ITER_SIZE not supported with Concrete DropBlock"
        old_max_iter = cfg.SOLVER.MAX_ITER
        iter_size = cfg.SOLVER.ITER_SIZE
        new_max_iter = old_max_iter * iter_size
        cfg.SOLVER.MAX_ITER = new_max_iter

        warnings.warn(f"SOLVER.ITER_SIZE is set to {iter_size}. "
                      f"MAX_ITER: {old_max_iter} -> {new_max_iter}. "
                      f"Scheduler will only be stepped every {iter_size} iterations "
                       "so other parameters can be kept unchanged.")

if __name__ == "__main__":
    main()
