import argparse
import os

from PIL import Image, ImageDraw

import numpy as np

import pykitti


def get_args():
    parser = argparse.ArgumentParser(
        description="MonoLayout DataPreparation options")
    parser.add_argument("--base_path", type=str, default="./kitti_raw",
                        help="Path to the root data directory")
    parser.add_argument("--date", type=str, default="2011_09_26",
                        help="Corresponding date from the KITTI RAW dataset")
    parser.add_argument(
        "--sequence",
        type=str,
        default="0001",
        help="Sequence number corresponding to a particular date")
    parser.add_argument("--out_dir", type=str, default='',
                        help="Output directory to save layouts")
    parser.add_argument(
        "--range",
        type=int,
        default=40,
        help="Size of the rectangular grid in metric space (in m)")
    parser.add_argument("--occ_map_size", type=int, default=256,
                        help="Occupancy map size (in pixels)")
    parser.add_argument(
        "--seg_class",
        type=str,
        choices=[
            "road",
            "sidewalk",
            "vehicle"],
        help="Data Preparation for Road/Sidewalk/Vehicle")
    parser.add_argument(
        "--process",
        type=str,
        choices=[
            "all",
            "one"],
        default="one",
        help="Process entire KITTI RAW dataset or one sequence at a time")

    return parser.parse_args()


def get_road_velo_pts(velo_pts, frame_seg_img, Tr_cam2_velo, K, req_labels):

    image_pts = np.matmul(K, np.matmul(Tr_cam2_velo, velo_pts))
    image_pts[0, :] = image_pts[0, :] / image_pts[2, :]
    image_pts[1, :] = image_pts[1, :] / image_pts[2, :]
    image_pts = np.array(image_pts[0:2, :], dtype=np.int32)
    img_rows = np.shape(frame_seg_img)[0]
    img_cols = np.shape(frame_seg_img)[1]

    flag1 = image_pts[0, :] > 0
    flag2 = image_pts[0, :] < img_cols
    flag3 = image_pts[1, :] > 0
    flag4 = image_pts[1, :] < img_rows
    flag = flag1 * flag2 * flag3 * flag4
    velo_pts = velo_pts[:, flag]
    image_pts = image_pts[:, flag]
    road_flag = np.zeros((np.shape(velo_pts)[1]), dtype=np.bool)
    for req_label in req_labels:
        road_flag = np.logical_or(
            road_flag,
            frame_seg_img[image_pts[1, :], image_pts[0, :]] == req_label)
    road_velo_pts = velo_pts[:, road_flag]

    return road_velo_pts


def get_rect(x, y, width, height, theta):
    rect = np.array([(-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2),
                     (-width / 2, -height / 2)])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset

    return transformed_rect


def get_3Dbox_to_2Dbox(label_path, length, width, res, out_dir):

    if label_path[-3:] != "txt":
        return

    TopView = np.zeros((int(length / res), int(width / res)))
    labels = open(label_path).read()
    labels = labels.split("\n")
    img = Image.fromarray(TopView)
    for label in labels:
        if label == "":
            continue

        elems = label.split()
        if elems[0] in ['Car', 'Van', 'Bus', 'Truck']:
            center_x = int(float(elems[11]) / res + width / (2 * res))
            center_z = int(float(elems[13]) / res)
            orient = -1 * float(elems[14])

            obj_w = int(float(elems[9]) / res)
            obj_l = int(float(elems[10]) / res)

            rectangle = get_rect(
                center_x, int(
                    length / res) - center_z, obj_l, obj_w, orient)
            draw = ImageDraw.Draw(img)
            draw.polygon([tuple(p) for p in rectangle], fill=255)

    img = img.convert('L')
    file_path = os.path.join(
        out_dir,
        os.path.basename(label_path)[
            :-3] + "png")
    img.save(file_path)
    print("Saved file at %s" % file_path)


def account_for_missing_files(args):
    velo_path = os.path.join(
        args.base_path,
        "2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/")
    os.system(
        "cp -r %s %s" %
        (os.path.join(
            velo_path, "0000000176.bin"), os.path.join(
            velo_path, "0000000177.bin")))
    os.system(
        "cp -r %s %s" %
        (os.path.join(
            velo_path, "0000000176.bin"), os.path.join(
            velo_path, "0000000178.bin")))
    os.system(
        "cp -r %s %s" %
        (os.path.join(
            velo_path, "0000000181.bin"), os.path.join(
            velo_path, "0000000179.bin")))
    os.system(
        "cp -r %s %s" %
        (os.path.join(
            velo_path, "0000000181.bin"), os.path.join(
            velo_path, "0000000180.bin")))


def get_static_bev(args, date, sequence):
    basedir = args.base_path
    if args.out_dir == "":
        out_dir = basedir
    seg_class = 'road'

    # Taking care of missing files in date 2011_09_26 sequence 0009
    if date == "2011_09_26" and sequence == "0009":
        account_for_missing_files(args)

    label = {'road': [13, 24], 'sidewalk': [15]}
    res = args.range / float(args.occ_map_size)
    xmin = - args.range / 2
    xmax = args.range / 2
    zmin = 0
    zmax = args.range
    rows = args.occ_map_size
    cols = args.occ_map_size

    data = pykitti.raw(args.base_path, date, sequence)

    num_frames = len(data.timestamps)
    print("Processing Date: %s, Sequence: %s" % (date, sequence))

    K = np.zeros((4, 4))
    K[0:3, 0:3] = data.calib.K_cam2
    K[3, 3] = 1
    Tr_velo_imu = data.calib.T_velo_imu  # to velodyne frame from imu frame
    Tr_cam2_velo = data.calib.T_cam2_velo  # to cam2 frame from velodyne frame

    oxts = data.oxts

    output_folder = os.path.join(
        out_dir, date, date + '_drive_' + sequence + '_sync', "%s_%d" %
        (seg_class, args.occ_map_size))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dense_class_base_pts_hom = np.zeros((4, 1))
    for frame_no in range(num_frames):

        # to base imu frame from current imu frame
        Tr_base_current_imu = oxts[frame_no].T_w_imu
        Tr_base_cam2 = np.matmul(
            Tr_base_current_imu,
            np.matmul(
                np.linalg.inv(Tr_velo_imu),
                np.linalg.inv(Tr_cam2_velo)))

        velo_pts = data.get_velo(frame_no)
        velo_pts = velo_pts[velo_pts[:, 0] > 0, :]
        velo_pts = velo_pts[velo_pts[:, 0] < 30, :]
        velo_pts_hom = np.ones((4, np.shape(velo_pts)[0]))
        velo_pts_hom[0:3, :] = velo_pts[:, 0:3].T
        frame_seg_path = os.path.join(
            args.base_path,
            date,
            date +
            '_drive_' +
            sequence +
            '_sync',
            "image_02/segmentation",
            str(frame_no).zfill(10) +
            '.png')
        frame_seg_img = np.asarray(Image.open(frame_seg_path))
        seg_class_velo_pts_hom = get_road_velo_pts(
            velo_pts_hom, frame_seg_img, Tr_cam2_velo, K, label[seg_class])
        seg_class_cam_pts_hom = np.matmul(Tr_cam2_velo, seg_class_velo_pts_hom)

        occupancy_map = np.zeros((rows, cols))
        flag1 = seg_class_cam_pts_hom[0, :] > xmin
        flag2 = seg_class_cam_pts_hom[0, :] < xmax
        flag3 = seg_class_cam_pts_hom[2, :] > zmin
        flag4 = seg_class_cam_pts_hom[2, :] < zmax
        flag = flag1 * flag2 * flag3 * flag4
        seg_class_cam_pts_hom = seg_class_cam_pts_hom[:, flag]
        seg_class_base_pts_hom = np.matmul(Tr_base_cam2, seg_class_cam_pts_hom)
        dense_class_base_pts_hom = np.concatenate(
            (dense_class_base_pts_hom, seg_class_base_pts_hom), axis=1)
        print("reading velodyne points from frame " + str(frame_no).zfill(6))

    for frame_no in range(num_frames):

        Tr_current_base_imu = np.linalg.inv(
            oxts[frame_no].T_w_imu)  # to current imu from base imu frame
        Tr_cam2_base = np.matmul(
            Tr_cam2_velo,
            np.matmul(
                Tr_velo_imu,
                Tr_current_base_imu))  # to cam2 frame from base imu frame
        dense_class_cam_pts_hom = np.matmul(
            Tr_cam2_base, dense_class_base_pts_hom)

        occupancy_map = np.zeros((rows, cols))
        flag1 = dense_class_cam_pts_hom[0, :] > xmin
        flag2 = dense_class_cam_pts_hom[0, :] < xmax
        flag3 = dense_class_cam_pts_hom[2, :] > zmin
        flag4 = dense_class_cam_pts_hom[2, :] < zmax
        flag = flag1 * flag2 * flag3 * flag4
        dense_class_cam_pts_hom = dense_class_cam_pts_hom[:, flag]
        plane_topview = np.zeros((2, np.shape(dense_class_cam_pts_hom)[1]))
        plane_topview[0, :] = (xmax + dense_class_cam_pts_hom[0, :]) / res
        plane_topview[1, :] = (zmax - dense_class_cam_pts_hom[2, :]) / res
        plane_topview = np.array(plane_topview, dtype=np.int32)
        occupancy_map[plane_topview[1, :], plane_topview[0, :]] = 255

        occ_map = Image.fromarray(occupancy_map)
        outpath = os.path.join(output_folder, str(frame_no).zfill(10) + '.png')
        occ_map = occ_map.convert('L')
        occ_map.save(outpath)
        print("generating bev for frame " + str(frame_no).zfill(6))


if __name__ == "__main__":
    args = get_args()
    if args.seg_class == "vehicle":
        out_dir = os.path.join(
            os.path.dirname(
                os.path.normpath(
                    args.base_path)),
            "vehicle_%d" %
            args.occ_map_size)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for file_path in os.listdir(args.base_path):
            label_path = os.path.join(args.base_path, file_path)
            get_3Dbox_to_2Dbox(
                label_path,
                args.range,
                args.range,
                args.range /
                float(
                    args.occ_map_size),
                out_dir)
    elif args.seg_class == "road":
        if args.process == "all":
            for date in os.listdir(args.base_path):
                for folder in os.listdir(os.path.join(args.base_path, date)):
                    if not os.path.isdir(
                        os.path.join(
                            args.base_path,
                            date,
                            folder)):
                        continue
                    sequence = folder.split("_")[-2]
                    get_static_bev(args, date, sequence)
        else:
            get_static_bev(args, args.date, args.sequence)
