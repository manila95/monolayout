
import argparse
import os

import monolayout

import numpy as np

import torch
from torch.utils.data import DataLoader

import tqdm

from utils import mean_IU, mean_precision


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation options")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")
    parser.add_argument("--pretrained_path", type=str, default="./models/",
                        help="Path to the pretrained model")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")
    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "argo",
            "3Dobject",
            "odometry",
            "raw"],
        help="Data split for training/validation")
    parser.add_argument("--ext", type=str, default="png",
                        help="File extension of the images")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "both",
            "static",
            "dynamic"],
        help="Type of model being trained")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="size of topview occupancy map")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of cpu workers for dataloaders")

    return parser.parse_args()


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def load_model(models, model_path):
    """Load model(s) from disk
    """
    model_path = os.path.expanduser(model_path)

    assert os.path.isdir(model_path), \
        "Cannot find folder {}".format(model_path)
    print("loading model from folder {}".format(model_path))

    for key in models.keys():
        print("Loading {} weights...".format(key))
        path = os.path.join(model_path, "{}.pth".format(key))
        model_dict = models[key].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {
            k: v for k,
            v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[key].load_state_dict(model_dict)
    return models


def evaluate():
    opt = get_args()

    # Loading Pretarined Model
    models = {}
    models["encoder"] = monolayout.Encoder(18, opt.height, opt.width, True)
    if opt.type == "both":
        models["static_decoder"] = monolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)
        models["dynamic_decoder"] = monolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)
    else:
        models["decoder"] = monolayout.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)

    for key in models.keys():
        models[key].to("cuda")

    models = load_model(models, opt.pretrained_path)

    # Loading Validation/Testing Dataset

    # Data Loaders
    dataset_dict = {"3Dobject": monolayout.KITTIObject,
                    "odometry": monolayout.KITTIOdometry,
                    "argo": monolayout.Argoverse,
                    "raw": monolayout.KITTIRAW}

    dataset = dataset_dict[opt.split]
    fpath = os.path.join(
        os.path.dirname(__file__),
        "splits",
        opt.split,
        "{}_files.txt")
    test_filenames = readlines(fpath.format("val"))
    test_dataset = dataset(opt, test_filenames, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        1,
        True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    iou, mAP = np.array([0., 0.]), np.array([0., 0.])
    for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)
        pred = np.squeeze(
            torch.argmax(
                outputs["topview"].detach(),
                1).cpu().numpy())
        true = np.squeeze(inputs[opt.type + "_gt"].detach().cpu().numpy())
        iou += mean_IU(pred, true)
        mAP += mean_precision(pred, true)
    iou /= len(test_loader)
    mAP /= len(test_loader)
    print("Evaluation Results: mIOU: %.4f mAP: %.4f" % (iou[1], mAP[1]))


def process_batch(opt, models, inputs):
    outputs = {}
    for key, input_ in inputs.items():
        inputs[key] = input_.to("cuda")

    features = models["encoder"](inputs["color"])

    if opt.type == "both":
        outputs["dynamic"] = models["dynamic_decoder"](features)
        outputs["static"] = models["static_decoder"](features)
    else:
        outputs["topview"] = models["decoder"](features)

    return outputs


if __name__ == "__main__":
    evaluate()
