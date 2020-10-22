
import argparse
import os

import monolayout

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import tqdm

from utils import mean_IU, mean_precision


def get_args():
    parser = argparse.ArgumentParser(description="MonoLayout options")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./models/",
                        help="Path to save models")
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        default="",
        help="Path to a pretrained model used for initialization")
    parser.add_argument("--model_name", type=str, default="monolayout",
                        help="Model Name with specifications")
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
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "both",
            "static",
            "dynamic"],
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
    parser.add_argument("--occ_map_size", type=int, default=128,
                        help="size of topview occupancy map")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Max number of training epochs")
    parser.add_argument("--log_frequency", type=int, default=5,
                        help="Log files every x epochs")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="Number of cpu workers for dataloaders")
    parser.add_argument("--lambda_D", type=float, default=0.01,
                        help="tradeoff weight for discriminator loss")
    parser.add_argument("--discr_train_epoch", type=int, default=5,
                        help="epoch to start training discriminator")
    parser.add_argument("--osm_path", type=str, default="./data/osm",
                        help="OSM path")

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
        self.models["encoder"] = monolayout.Encoder(
            18, self.opt.height, self.opt.width, True)
        if self.opt.type == "both":
            self.models["static_decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["static_discr"] = monolayout.Discriminator()
            self.models["dynamic_decoder"] = monolayout.Discriminator()
            self.models["dynamic_decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
        else:
            self.models["decoder"] = monolayout.Decoder(
                self.models["encoder"].resnet_encoder.num_ch_enc)
            self.models["discriminator"] = monolayout.Discriminator()

        for key in self.models.keys():
            self.models[key].to(self.device)
            if "discr" in key:
                self.parameters_to_train_D += list(
                    self.models[key].parameters())
            else:
                self.parameters_to_train += list(self.models[key].parameters())

        # Optimization
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        self.model_optimizer_D = optim.Adam(
            self.parameters_to_train_D, self.opt.lr)
        self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(
            self.model_optimizer_D, self.opt.scheduler_step_size, 0.1)

        self.patch = (1, self.opt.occ_map_size // 2 **
                      4, self.opt.occ_map_size // 2**4)

        self.valid = Variable(
            torch.Tensor(
                np.ones(
                    (self.opt.batch_size,
                     *self.patch))),
            requires_grad=False).float().cuda()
        self.fake = Variable(
            torch.Tensor(
                np.zeros(
                    (self.opt.batch_size,
                     *self.patch))),
            requires_grad=False).float().cuda()

        # Data Loaders
        dataset_dict = {"3Dobject": monolayout.KITTIObject,
                        "odometry": monolayout.KITTIOdometry,
                        "argo": monolayout.Argoverse,
                        "raw": monolayout.KITTIRAW}

        self.dataset = dataset_dict[self.opt.split]
        fpath = os.path.join(
            os.path.dirname(__file__),
            "splits",
            self.opt.split,
            "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        self.val_filenames = val_filenames
        self.train_filenames = train_filenames

        train_dataset = self.dataset(self.opt, train_filenames)
        val_dataset = self.dataset(self.opt, val_filenames, is_train=False)

        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)
        self.val_loader = DataLoader(
            val_dataset,
            1,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        if self.opt.load_weights_folder != "":
            self.load_model()

        print("Using split:\n  ", self.opt.split)
        print(
            "There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset),
                len(val_dataset)))

    def train(self):

        for self.epoch in range(self.opt.num_epochs):
            loss = self.run_epoch()
            print("Epoch: %d | Loss: %.4f | Discriminator Loss: %.4f" %
                  (self.epoch, loss["loss"], loss["loss_discr"]))

            if self.epoch % self.opt.log_frequency == 0:
                self.validation()
                self.save_model()

    def process_batch(self, inputs, validation=False):
        outputs = {}
        for key, inpt in inputs.items():
            inputs[key] = inpt.to(self.device)

        features = self.models["encoder"](inputs["color"])

        if self.opt.type == "both":
            outputs["dynamic"] = self.models["dynamic_decoder"](features)
            outputs["static"] = self.models["static_decoder"](features)
        else:
            outputs["topview"] = self.models["decoder"](features)
        if validation:
            return outputs
        losses = self.compute_losses(inputs, outputs)
        losses["loss_discr"] = torch.zeros(1)

        return outputs, losses

    def run_epoch(self):
        self.model_optimizer.step()
        self.model_optimizer_D.step()
        loss = {}
        loss["loss"], loss["loss_discr"] = 0.0, 0.0
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.train_loader)):
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            fake_pred = self.models["discriminator"](outputs["topview"])
            real_pred = self.models["discriminator"](inputs["discr"].float())
            loss_GAN = self.criterion_d(fake_pred, self.valid)
            loss_D = self.criterion_d(
                fake_pred, self.fake) + self.criterion_d(real_pred, self.valid)
            loss_G = self.opt.lambda_D * loss_GAN + losses["loss"]

            # Train Discriminator
            if self.epoch > self.opt.discr_train_epoch:
                loss_G.backward(retain_graph=True)
                self.model_optimizer_D.zero_grad()
                loss_D.backward()
                self.model_optimizer.step()
                self.model_optimizer_D.step()
            else:
                losses["loss"].backward()
                self.model_optimizer.step()

            loss["loss"] += losses["loss"].item()
            loss["loss_discr"] += loss_D.item()
        loss["loss"] /= len(self.train_loader)
        loss["loss_discr"] /= len(self.train_loader)
        return loss

    def validation(self):
        iou, mAP = np.array([0., 0.]), np.array([0., 0.])
        for batch_idx, inputs in tqdm.tqdm(enumerate(self.val_loader)):
            with torch.no_grad():
                outputs = self.process_batch(inputs, True)
            pred = np.squeeze(
                torch.argmax(
                    outputs["topview"].detach(),
                    1).cpu().numpy())
            true = np.squeeze(
                inputs[self.opt.type + "_gt"].detach().cpu().numpy())
            iou += mean_IU(pred, true)
            mAP += mean_precision(pred, true)
        iou /= len(self.val_loader)
        mAP /= len(self.val_loader)
        print(
            "Epoch: %d | Validation: mIOU: %.4f mAP: %.4f" %
            (self.epoch, iou[1], mAP[1]))

    def compute_losses(self, inputs, outputs):
        losses = {}
        if self.opt.type == "both":
            losses["static_loss"] = self.compute_topview_loss(
                                            outputs["static"],
                                            inputs["static"],
                                            self.weight[self.opt.type])
            losses["dynamic_loss"] = self.compute_topview_loss(
                                            outputs["dynamic_loss"],
                                            inputs["dynamic"],
                                            self.weight[self.opt.type])
        else:
            losses["loss"] = self.compute_topview_loss(
                                            outputs["topview"],
                                            inputs[self.opt.type],
                                            self.weight[self.opt.type])

        return losses

    def compute_topview_loss(self, outputs, true_top_view, weight):

        generated_top_view = outputs
        true_top_view = torch.squeeze(true_top_view.long())
        loss = nn.CrossEntropyLoss(weight=torch.Tensor([1., weight]).cuda())
        output = loss(generated_top_view, true_top_view)
        return output.mean()

    def save_model(self):
        save_path = os.path.join(
            self.opt.save_path,
            self.opt.model_name,
            self.opt.split,
            "weights_{}".format(
                self.epoch))

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
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print(
            "loading model from folder {}".format(
                self.opt.load_weights_folder))

        for key in self.models.keys():
            print("Loading {} weights...".format(key))
            path = os.path.join(
                self.opt.load_weights_folder,
                "{}.pth".format(key))
            model_dict = self.models[key].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[key].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(
            self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
