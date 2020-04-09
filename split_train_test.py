import os
import sys


file_name = sys.argv[2]
file_path = sys.argv[1]

f = open(file_name, "w")
for folder in os.listdir(file_path)[:8]:
    folder_path = os.path.join(file_path, folder, "road_dense128")
    for file_ in os.listdir(folder_path):
        f.write(os.path.join(folder, "road_dense128", file_))
        f.write("\n")


