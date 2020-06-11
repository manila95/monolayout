import argparse
import glob
import os

import PIL.Image as pil

import cv2

from monolayout import model

import numpy as np

import torch

from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser(
        description="Testing arguments for MonoLayout")
    parser.add_argument("--image_path", type=str,
                        help="path to folder of images", required=True)
    parser.add_argument("--model_path", type=str,
                        help="path to MonoLayout model", required=True)
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="extension of images in the folder")
    parser.add_argument("--out_dir", type=str,
                        default="output directory to save topviews")
    parser.add_argument("--type", type=str,
                        default="static/dynamic/both")

    return parser.parse_args()


def save_topview(idx, tv, name_dest_im):
    tv_np = tv.squeeze().cpu().numpy()
    true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
    true_top_view[tv_np[1] > tv_np[0]] = 255
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, true_top_view)

    print("Saved prediction to {}".format(name_dest_im))


def test(args):
    models = {}
    device = torch.device("cuda")
    encoder_path = os.path.join(args.model_path, "encoder.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]
    models["encoder"] = model.Encoder(18, feed_width, feed_height, False)
    filtered_dict_enc = {
        k: v for k,
        v in encoder_dict.items() if k in models["encoder"].state_dict()}
    models["encoder"].load_state_dict(filtered_dict_enc)

    if args.type == "both":
        static_decoder_path = os.path.join(
            args.model_path, "static_decoder.pth")
        dynamic_decoder_path = os.path.join(
            args.model_path, "dynamic_decoder.pth")
        models["static_decoder"] = model.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)
        models["static_decoder"].load_state_dict(
            torch.load(static_decoder_path, map_location=device))
        models["dynamic_decoder"] = model.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)
        models["dynamic_decoder"].load_state_dict(
            torch.load(dynamic_decoder_path, map_location=device))
    else:
        decoder_path = os.path.join(args.model_path, "decoder.pth")
        models["decoder"] = model.Decoder(
            models["encoder"].resnet_encoder.num_ch_enc)
        models["decoder"].load_state_dict(
            torch.load(decoder_path, map_location=device))

    for key in models.keys():
        models[key].to(device)
        models[key].eval()

    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(
            args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.out_dir
        try:
            os.mkdir(output_directory)
        except BaseException:
            pass
    else:
        raise Exception(
            "Can not find args.image_path: {}".format(
                args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize(
                (feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = models["encoder"](input_image)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            print(
                "Processing {:d} of {:d} images- ".format(idx + 1, len(paths)))
            if args.type == "both":
                static_tv = models["static_decoder"](
                    features, is_training=False)
                dynamic_tv = models["dynamic_decoder"](
                    features, is_training=False)
                save_topview(
                    idx,
                    static_tv,
                    os.path.join(
                        args.out_dir,
                        "static",
                        "{}.png".format(output_name)))
                save_topview(
                    idx,
                    dynamic_tv,
                    os.path.join(
                        args.out_dir,
                        "dynamic",
                        "{}.png".format(output_name)))
            else:
                tv = models["decoder"](features, is_training=False)
                save_topview(
                    idx,
                    tv,
                    os.path.join(
                        args.out_dir,
                        args.type,
                        "{}.png".format(output_name)))

    print('-> Done!')


if __name__ == "__main__":
    args = get_args()
    test(args)
