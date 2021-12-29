import argparse
import cv2
import math
import os

def viz_dataset_side_by_side(img_folder, target_folder, start = 0, end = math.inf, keep_records = False):
    count = start
    records = set() # data instace to keep for wtver reason
    while True:
        #paths
        img_path = os.path.join(img_folder, str(count) + ".jpg")
        target_path = os.path.join(target_folder, str(count) + ".jpg")
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        target = cv2.resize(target, dsize = (1654,1170))
        # concatenate images and resize
        numpy_horizontal_concat = cv2.hconcat([img, target])
        numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat,
                                             dsize = (1600, 800),
                                             interpolation = cv2.INTER_CUBIC)
        #visualize
        text ="n for next, r for recordn, p for previous, q to quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(numpy_horizontal_concat, text, (20, 20), font, 1, (0, 255, 0), 3)
        cv2.imshow('Image ' + str(count), numpy_horizontal_concat)
        key = cv2.waitKey()
        if key == ord('n'):
            count += 1
            cv2.destroyAllWindows()
        elif key == ord('p'):
            count -= 1
            cv2.destroyAllWindows()
        elif key == ord('r'):
            records.add(count)
            cv2.destroyAllWindows()
        elif key == ord('q') or count == end:
            break
    #do sum with records


if __name__ == "__main__":
    """
    Exemple:
    python3 viz_dataset.py /home/babacar/cours_CS/3A/DeepL/project/manga_translation/\
    data/final/images /home/babacar/cours_CS/3A/DeepL/project/manga_translation/data/\
    final/target 0 100 0
    """
    parser = argparse.ArgumentParser(description = 'Visualize images and targets')
    parser.add_argument('img_folder', help='path to images. format i.jpg')
    parser.add_argument('target_folder', help='pathto targets. format i.jpg')
    parser.add_argument('start', type = int, help = "Index of the first image to show")
    parser.add_argument('end', type = int, help = "Index of the last image to show")
    parser.add_argument('keep_records', type = bool, help = "Save records in \
        text file or not")
    args = parser.parse_args()

    viz_dataset_side_by_side(args.img_folder,
                             args.target_folder,
                             args.start,
                             args.end,
                             args.keep_records)