import argparse
import os

from shutil import copyfile


def flatten_image_folder(input_path, target_path, bboxes_path, target_output_path):
    if not os.path.exists(target_output_path):
        os.mkdir(target_output_path)
    weak_set = os.path.join(target_output_path, "weak")
    full_set = os.path.join(target_output_path, "full")
    if not os.path.exists( os.path.join(target_output_path, "weak")):
        os.mkdir(weak_set)
        os.mkdir(full_set)
    for path in [weak_set, full_set]:
        if not os.path.exists(os.path.join(path, "images")):
            os.mkdir(os.path.join(path, "images"))
            os.mkdir(os.path.join(path, "masks"))
            os.mkdir(os.path.join(path, "bboxes"))

    counter = 0
    for folder in sorted(os.listdir(input_path)):
        print("Processing for manga ", folder, " ...")
        for img_name in sorted(os.listdir(os.path.join(input_path, folder))):
            counter += 1
            if not  os.path.exists(os.path.join(target_path, folder, img_name.replace('jpg', 'png').zfill(3))):
                #processes in weak set
                # origin path
                img_path = os.path.join(
                    input_path, folder, img_name.zfill(3))
                target_img_path = os.path.join(target_path, folder, img_name.replace('jpg', 'png'))
                bbox_img_path = os.path.join(bboxes_path, folder, img_name)
                # destination path
                output_img_path = os.path.join(weak_set,"images", str(counter) + ".jpg")
                bbox_target_path = os.path.join(
                    weak_set, "bboxes", str(counter) + ".jpg")



                #copy files
                copyfile(img_path, output_img_path)
                copyfile(bbox_img_path, bbox_target_path)
                print(counter, " images processed...")
            else:
                #processes in full set
                # origin path
                img_path = os.path.join(
                    input_path, folder, img_name.zfill(3))
                target_img_path = os.path.join(target_path, folder, img_name.replace('jpg', 'png'))
                bbox_img_path = os.path.join(bboxes_path, folder, img_name)
                # destination path
                output_img_path = os.path.join(full_set, "images", str(counter) + ".jpg")
                target_output_img_path = os.path.join(
                    full_set, "masks", str(counter) + ".jpg")
                bbox_target_path = os.path.join(
                    full_set, "bboxes", str(counter) + ".jpg")
                #copy files
                copyfile(img_path, output_img_path)
                copyfile(target_img_path, target_output_img_path)
                copyfile(bbox_img_path, bbox_target_path)
                print(counter, " images processed...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare image folder \
            and target folder for training. Each folder contains \
            image entitled i.jpg \n \
    ex: python3 ../tools/prepare_data.py ./raw/images ./raw/masks ./raw/bboxes/ /final/ ')
    parser.add_argument(
        'input', help='path raw manga109 images (folder of folders)')
    parser.add_argument(
        'mask', help='path segmented target images (folder of folders)')
    parser.add_argument('bboxes', help='path to bounding boxes')
    parser.add_argument(
        'target_output', help='path where target images will be stored')
    args = parser.parse_args()

    flatten_image_folder(args.input,
                         args.mask,
                         args.bboxes,
                         args.target_output)
