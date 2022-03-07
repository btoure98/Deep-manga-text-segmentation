import argparse
import cv2
import os

from xml_process import get_frames


def draw_bboxes(img, boxes):
    for rect in boxes:
        cv2.rectangle(img, (int(rect[0]), int(rect[1])),
                      (int(rect[2]), int(rect[3])), 255, -1)
    return img


def create_bbox_images(manga, annotations_path, output_path):
    # creation of folders and stuff
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if len(os.listdir(output_path)) != 0:
        print("folder not empty: " + output_path)
        return
    # position of texts
    annotations = get_frames(annotations_path)
    for image_name in sorted(os.listdir(manga)):
        # read image
        image_path = os.path.join(manga, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img[:, :] = 0  # all black image
        # get annotation
        annotation = annotations[int(image_name.split(".")[0])]
        # segment bounding boxes
        segmented_image = draw_bboxes(img, annotation)
        cv2.imwrite(os.path.join(output_path, image_name), segmented_image)


def main(mangas, output):
    if not os.path.exists(output):
        os.mkdir(output)
    image_folders_path = os.path.join(mangas, "images")
    annotations_path = os.path.join(mangas, "annotations")
    for manga in sorted(os.listdir(image_folders_path)):
        print("Processing for ", manga, )
        manga_path = os.path.join(image_folders_path, manga)
        annotation = os.path.join(annotations_path, manga + ".xml")
        output_manga = os.path.join(output, manga)
        print(manga_path)
        print(annotation)
        print(output_manga)
        create_bbox_images(manga_path, annotation, output_manga)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creation of bounding boxes')
    parser.add_argument('mangas', help='path to data: mangas and annotation')
    parser.add_argument(
        'output', help='path where target images will be stored')
    args = parser.parse_args()

    main(args.mangas, args.output)
