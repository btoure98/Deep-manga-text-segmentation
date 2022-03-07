import argparse
import cv2
import math
import os

x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False


def mouse_crop(event, x, y, param):
    global x_start, y_start, x_end, y_end, cropping
    count = param[0]
    dest = param[1]
    img = param[2]
    target = param[3]
    h = img.shape[0]
    w = img.shape[1]
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished
        refPoint = [(int((x_start - w//2)*2), int(y_start)),
                    (int((x_end - w//2)*2), int(y_end))]

        if len(refPoint) == 2:  # when two points were found
            print(refPoint[0][1], refPoint[1][1],
                  refPoint[0][0], refPoint[1][0])
            cropped_img = img[refPoint[0][1]:refPoint[1]
                              [1], refPoint[0][0]:refPoint[1][0]]
            cropped_target = target[refPoint[0][1]
                :refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imwrite(os.path.join(dest, "images",
                        str(count) + ".jpg"), cropped_img)
            cv2.imwrite(os.path.join(dest, "target", str(
                count) + ".jpg"), cropped_target)
            print("finished cropping")


def crop_images(img_folder, target_folder, start=0, end=math.inf, destination="./"):
    count = start
    for count in range(start, end + 1):
        # paths
        img_path = os.path.join(img_folder, str(count) + ".jpg")
        target_path = os.path.join(target_folder, str(count) + ".jpg")
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        """
        img = cv2.resize(img,
                         dsize = (585, 827),
                         interpolation = cv2.INTER_CUBIC)
        target = cv2.resize(target,
                   dsize = (585, 827),
                   interpolation = cv2.INTER_CUBIC)
        """
        # concatenate images and resize
        numpy_horizontal_concat = cv2.hconcat([img, target])
        numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat,
                                             dsize=(1654, 1170),
                                             interpolation=cv2.INTER_CUBIC)
        ################################
        ################################
        cv2.imshow('Image ' + str(count), numpy_horizontal_concat)
        # cv2.imshow(numpy_horizontal_concat)
        cv2.setMouseCallback('Image ' + str(count), mouse_crop,
                             [count, destination, img, target])
        while True:
            i = numpy_horizontal_concat.copy()
            if not cropping:
                cv2.imshow('Image ' + str(count), i)

            elif cropping:
                cv2.rectangle(i, (x_start, y_start),
                              (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow('Image ' + str(count), i)
            if cv2.waitKey(1) == ord('q'):
                exit()
            elif cv2.waitKey(1) == ord('n'):
                break
        # close all open windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize images and targets')
    parser.add_argument('img_folder', help='path to images. format i.jpg')
    parser.add_argument('target_folder', help='pathto targets. format i.jpg')
    parser.add_argument('start', type=int,
                        help="Index of the first image to show")
    parser.add_argument(
        'end', type=int, help="Index of the last image to show")
    parser.add_argument(
        'destination', help="destination folder of cropped images")
    args = parser.parse_args()

    crop_images(args.img_folder,
                args.target_folder,
                args.start,
                args.end,
                args.destination)
