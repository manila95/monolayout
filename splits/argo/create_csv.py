import os
import csv

dataset_dir = "/scratch/shubhika/datasets/ARGO/argoverse-tracking/"
folder_paths = os.listdir(dataset_dir)

train = []
val = []

for folder in folder_paths:
    print(folder)

    if os.path.isfile(os.path.join(dataset_dir, folder)):
        continue

    sub_folder_path = os.listdir(os.path.join(dataset_dir, folder))

    for sub_folder in sub_folder_path:
        paths = os.listdir(
            os.path.join(dataset_dir, folder, sub_folder, "stereo_front_left")
        )
        print(sub_folder)

        for idx, image_path in enumerate(paths):

            full_img_path = os.path.join(
                dataset_dir, folder, sub_folder, "stereo_front_left", image_path
            )
            full_gt_path = os.path.join(
                dataset_dir, folder, sub_folder, "gt_top", image_path
            )
            curr_img = [full_img_path, full_gt_path]
            if folder == "train4":
                val.append(curr_img)
            else:
                train.append(curr_img)

with open("argo_train.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(train)

csvFile.close()

with open("argo_val.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(val)

csvFile.close()
