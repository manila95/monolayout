
import numpy as np
import time
# import eval_segm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from validate import *
import os
import json
import tqdm
import argparse
# from utils import *
# from kitti_utils import *
# from layers import *
# from metric.iou import IoU
# from fcn_iou import *
from IPython import embed
from torch.autograd import Variable


import model
import dataloader

def get_args():
    parser 	= argparse.ArgumentParser(description="MonoLayout options")
    parser.add_argument("--data_path", type=str, default="./data",
                         help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./models/",
                         help="Path to save models")
    parser.add_argument("--model_name", type=str, default="monolayout",
                         help="Model Name with specifications")
    parser.add_argument("--split", type=str, choices=["argo", "3Dobject", "odometry"],
                         help="Data split for training/validation")
    parser.add_argument("--ext", type=str, default="png",
                         help="File extension of the images")
    parser.add_argument("--height", type=int, default=512,
                         help="Image height")
    parser.add_argument("--width", type=int, default=512,
                         help="Image width")
    parser.add_argument("--type", type=str, choices=["both", "static", "dynamic"],
                         help="Type of model being trained")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Mini-Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="learning rate")
    parser.add_argument("--lr_D", type=float, default=1e-5,
                         help="discriminator learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=5,
                         help="step size for the both schedulers")
    parser.add_argument("--static_weight", type=float, default=5.,
                         help="static weight for calculating loss")
    parser.add_argument("--dynamic_weight", type=float, default=15.,
                         help="dynamic weight for calculating loss")
    parser.add_argument("--num_epochs", type=int, default=100,
                         help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,
                         help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=12,
                         help="Number of cpu workers for dataloaders")

    return parser.parse_args()


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Trainer:
    def __init__(self):
        self.opt = get_args()
        self.models = {}
        self.weight = {}
        self.weight["static"] = self.opt.static_weight
        self.weight["dynamic"] = self.opt.dynamic_weight
        self.device = "cuda"
        self.criterion_d = nn.BCEWithLogitsLoss()
        self.parameters_to_train = []
        self.parameters_to_train_D = []

		# Initializing models
        self.models["encoder"] = model.Encoder(18, self.opt.height, self.opt.width, True)
        if self.opt.type == "both":
            self.models["static_decoder"] = model.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["static_discr"] = model.Discriminator()
            self.models["dynamic_decoder"] = model.Discriminator()
            self.models["dynamic_decoder"] = model.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
        else:
            self.models["decoder"] = model.Decoder(self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["discriminator"] = model.Discriminator(2)

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "discr" in key:
                self.parameters_to_train_D += list(self.models[key].parameters())
            else:
                self.parameters_to_train += list(self.models[key].parameters())

        # Optimization 
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 
            self.opt.scheduler_step_size, 0.1)

        self.model_optimizer_D = optim.Adam(self.parameters_to_train_D, self.opt.lr)
        self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(self.model_optimizer_D, 
            self.opt.scheduler_step_size, 0.1)


        ## Data Loaders
        dataset_dict = {"3Dobject": dataloader.KITTIObject, 
                        "odometry": dataloader.KITTIOdometry,
                        "argo":     dataloader.Argoverse}

        self.dataset = dataset_dict[self.opt.split]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames
        img_ext = '.png' if self.opt.ext=="png" else '.jpg'


        train_dataset = self.dataset(self.opt, train_filenames)
        val_dataset = self.dataset(self.opt, val_filenames)

        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True,
                                       num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True,
                                       num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)


        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))


    def train(self):

        for self.epoch in range(self.opt.num_epochs):
            loss = self.run_epoch()
            print("Epoch: %d | Loss: %.4f | Discriminator Loss: %.4f"%(self.epoch, loss["loss"], 
                loss["loss_discr"]))

            if self.epoch % self.opt.log_frequency:
                self.save_model()

    def run_epoch(self):
        self.model_optimizer.step()
        # self.model_optimizer_D.step()
        loss = {}
        loss["loss"], loss["loss_discr"] = 0., 0.
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            loss["loss"] += losses["loss"].item()
            loss["loss_discr"] += losses["loss_discr"].item()
        loss["loss"] /= len(self.train_loader)
        loss["loss_discr"] /= len(self.train_loader)
        return loss


    def process_batch(self, inputs):
        outputs = {}
        for key, inpt in inputs.items():
            inputs[key] = inpt.to(self.device)

        features = self.models["encoder"](inputs["color"])
        
        if self.opt.type == "both":
            outputs["dynamic"] = self.models["dynamic_decoder"](features)
            outputs["static"] = self.models["static_decoder"](features)
            losses["loss"] = losses["dynamic_loss"] + losses["static_loss"]
            losses["loss_discr"] = torch.zeros(1)
        else:
            outputs["topview"] = self.models["decoder"](features)

        losses = self.compute_losses(inputs, outputs)
        losses["loss_discr"] = torch.zeros(1)

        return outputs, losses


    def compute_losses(self, inputs, outputs):
        losses = {}
        if self.opt.type == "both":
            losses["static_loss"] = self.compute_topview_loss(outputs["static"], inputs["static"], 
                self.weight[self.opt.type])
            losses["dynamic_loss"] = self.compute_topview_loss(outputs["dynamic_loss"], inputs["dynamic"],
                self.weight[self.opt.type])
        else:
            losses["loss"] = self.compute_topview_loss(outputs["topview"], inputs[self.opt.type], 
                self.weight[self.opt.type])

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):

        generated_top_view = outputs
        #true_top_view = torch.ones(generated_top_view.size()).cuda()
        #loss = self.weighted_binary_cross_entropy(generated_top_view, true_top_view, torch.Tensor([1, 25]))
        true_top_view = torch.squeeze(true_top_view.long())
        loss = nn.CrossEntropyLoss(weight = torch.Tensor([1., weight]).cuda())
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def save_model(self):
        save_path = os.path.join(self.opt.save_path, self.opt.model_name, "weights_{}".format(self.epoch))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, "{}.pth".format(model_name))
            state_dict = model.state_dict()
            if model_name == "encoder":
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width

            torch.save(state_dict, model_path)
        optim_path = os.path.join(save_path, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def load_model(self):
        pass



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()