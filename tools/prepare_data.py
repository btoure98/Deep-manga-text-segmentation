import argparse
import cv2
import os

from shutil import copyfile

def flatten_image_folder(input_path, target_path, output_path, target_output_path):
    if not (os.path.exists(target_path) and os.path.exists(target_output_path)):
        os.mkdir(output_path)
        os.mkdir(target_output_path)
    counter = 0
    for folder in sorted(os.listdir(target_path)):
        print("Processing for manga ", folder, " ...")
        for img_name in sorted(os.listdir(os.path.join(target_path, folder))):
            #origin path
            img_path = os.path.join(input_path, folder, img_name.replace('png', 'jpg').zfill(3))
            target_img_path = os.path.join(target_path, folder, img_name)
            #destination path
            output_img_path = os.path.join(output_path, str(counter) + ".jpg")
            target_output_img_path = os.path.join(target_output_path, str(counter) + ".jpg")
            if not os.path.exists(target_img_path):
                exit()
            #actually copy files
            copyfile(img_path, output_img_path)
            copyfile(target_img_path, target_output_img_path)
            counter += 1
            print(counter, " images processed...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Prepare image folder \
            and target folder for training. Each folder contains \
            image entitled i.jpg \n \
    ex: python3 prepare_data.py ./raw/images ./segmented_text ./final/images ./final/target ')
    parser.add_argument('input', help='path raw manga109 images (folder of folders)')
    parser.add_argument('target', help='path segmented target images (folder of folders)')
    parser.add_argument('output', help='path where images will be stored')
    parser.add_argument('target_output', help='path where target images will be stored')
    args = parser.parse_args()

    flatten_image_folder(args.input,
                         args.target,
                         args.output,
                         args.target_output)