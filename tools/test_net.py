import argparse
import os
import sys
import torch

o_path = os.getcwd()
sys.path.append(o_path)

from configs import cfg
from utils.logger import setup_logger
from utils.comm import synchronize, get_rank
from utils.collect_env import collect_env_info
from modeling.detector.detectors import build_detection_model

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "--task",
        default="det",
        type=str,
        help="eval task: det | corloc",
    )
    parser.add_argument(
        "--vis",
        dest="vis",
        help="Visualize the final results",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = "./log/"
    logger = setup_logger("wetectron", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    model = build_detection_model(cfg)
    # model.to(cfg.MODEL.DEVICE)

if __name__ == "__main__":
    main()
