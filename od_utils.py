"""
A module of utils to assist in object detection using
the Faster-RCNN algorithms. In this case, only the
region proposal network is needed since classification
is not necessary beyond binary foreground / background.

Author: Simon Thomas
Data : 26/03/19
Updated: 27/03/19

References:
    code:
        - https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras/blob/master/frcnn_train_vgg.ipynb
    original paper:
        - https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf
"""
import numpy as np
import tensorflow as tf
import keras.backend as K

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as BBox

from skimage import io


# --- COLORS --- #
green = [0.36, 0.96, 0.26]


def read_annotation_file(annotation_dir, image_dir, file):
    """
    Parse the data from annotation file and get metadata from image

    Example Annotation csv file - does not include header
    # BX,BY,Width,Height
    24,23,120,139
    161,7,102,131
    273,69,108,126
    """
    annotation_file = os.path.join(annotation_dir, file) + ".csv"
    bounding_boxes = np.genfromtxt(annotation_file, delimiter=',')

    image_file = os.path.join(image_dir, file) + ".png"
    image = io.imread(image_file)

    cols, rows = image.shape[:2]

    return {"filepath": image_file, "width": cols, "height": rows, "bboxes": bounding_boxes}


class DataGenerator(object):
    """
    A data generator that gives a train and testing batch.

    ! - This is for 200x400 images only -!
    """
    def __init__(self, annotation_dir, image_dir, num_anchors=9):
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.files = [file.split(".")[0] for file in os.listdir(image_dir)]
        self.mode = "train"
        self.batch = None
        self.im_rows = 200
        self.im_cols = 400
        self.out_rows = 7
        self.out_cols = 13
        self.num_anchors = num_anchors
        self.anchors = [
                        (64, 64), (128, 64), (64, 128),
                        (128, 128), (256, 128), (128, 256),
                        (256, 256), (512, 256), (256, 512)
                        ][0:num_anchors]  # only grab needed
        self.plot = False
        self.colors = ["red", "green", "blue"]


    def create_batch(self):
        # Containers for batch
        objects = []
        regressions = []
        images = []

        # Training and Test mode can produce different outputs?
        if self.mode == 'train':

            # Calculate RPN for each image
            for file in self.files:
                # Load annotation
                data = read_annotation_file(self.annotation_dir, self.image_dir, file)
                # Load image - without alpha
                image = io.imread(os.path.join(self.image_dir, file + ".png")).astype("float32")[:, :, 0:3]

                if self.plot:
                    plt.imshow(image)
                    ax = plt.gca()

                # Get bs i.e ground truth boxes
                bs = []
                for box in data["bboxes"]:
                    x1, y1, x2, y2 = box[0], box[1], box[0]+box[2], box[1]+box[3]
                    bs.append([x1, y1, x2, y2])

                    if self.plot:
                        # Add to plot
                        x, y = x1, y1
                        width = box[2]
                        height = box[3]
                        bbox = BBox((x, y), width, height, linewidth=2, edgecolor='yellow', facecolor='none')
                        ax.add_patch(bbox)

                # Get a - rpn boxes
                anchor_row = 0
                anchor_col = 0
                best_anchor_for_gt = {}

                # Step sizes
                row_step_size = self.im_rows // self.out_rows
                col_step_size = self.im_cols // self.out_cols

                for row in range(0, self.im_rows - row_step_size , row_step_size):
                    for col in range(0, self.im_cols - col_step_size, col_step_size):
                        # For every anchor at position of proposal

                        # Default all anchor regions as background
                        best_anchor_for_gt[anchor_row, anchor_col] = {"type": "neg"}

                        # anchor, bbox iou
                        best_iou_for_loc = [None, None, 0.0]

                        for i in range(self.num_anchors):
                            width, height = self.anchors[i]

                            # Top of anchor box
                            y, x = row-(height//2), col-(width//2)
                            a = [x, y, x+height, y+width]

                            # Ignore boxes that go across image boundaries
                            if a[0] < 0 or a[1] < 0 or a[2] > self.im_cols or a[3] > self.im_rows:
                                continue

                            # Check IoU for each bounding box in proposal:
                            for box_num, b in enumerate(bs):
                                IoU = iou(a, b)
                                if IoU > best_iou_for_loc[2]:
                                    best_iou_for_loc[0] = a
                                    best_iou_for_loc[1] = b
                                    best_iou_for_loc[2] = IoU

                            # Found best for location
                            if best_iou_for_loc[0]:

                                if best_iou_for_loc[2] > 0.5:

                                    # Set as foreground i.e. Yes, there is an object
                                    best_anchor_for_gt[anchor_row, anchor_col]["type"] = "pos"
                                    best_anchor_for_gt[anchor_row, anchor_col][i] = {}

                                    # Calculate offests
                                    # i.e. ∆ x−centre , ∆ y−centre , ∆ width , ∆ height for each anchor
                                    # get center coords of gt
                                    b = best_iou_for_loc[1]
                                    width_bb = b[2]-b[0]
                                    height_bb = b[3]-b[1]
                                    x_bb = b[0] + (width_bb/2)
                                    y_bb = b[1] + (height_bb/2)

                                    # get center coords of anchor
                                    a = best_iou_for_loc[0]
                                    width_a = a[2] - a[0]
                                    height_a = a[3] - a[1]
                                    x_a = anchor_col * (self.im_cols // self.out_cols)
                                    y_a = anchor_row * (self.im_rows // self.out_rows)

                                    # find deltas
                                    delta_x = x_a - x_bb
                                    delta_y = y_a - y_bb
                                    delta_w = width_a - width_bb
                                    delth_h = height_a - height_bb

                                    best_anchor_for_gt[anchor_row, anchor_col][i]["offset"] = [
                                                                                    delta_x,
                                                                                    delta_y,
                                                                                    delta_w,
                                                                                    delth_h ]

                                    # ---------------------------------------------------------------#
                                    if self.plot:
                                        # Plot bbox
                                        x, y, width, height = a[0], a[1], a[2]-a[0], a[3]-a[1]
                                        bbox = BBox((x, y), width, height,
                                                    linewidth=1,
                                                    edgecolor=self.colors[i%3],
                                                    facecolor='none')

                                        # Add the patch to the Axes
                                        ax.add_patch(bbox)

                                        # Plot center points
                                        ax.scatter(x_a, y_a, color="purple", s=10)
                                        ax.scatter(x_bb, y_bb, color="yellow", s=10)
                                    # --------------------------------------------------------------#


                                elif best_iou_for_loc[2] > 0.1:
                                    # Ambiguous since 0.1 < IoU < 0.5
                                    best_anchor_for_gt[anchor_row, anchor_col]["type"] = "neutral"




                        # inner
                        anchor_col += 1
                    # outer
                    anchor_col = 0
                    anchor_row += 1
                # ------------------------------------- END OF BIG LOOP ------------------------------------ #

                if self.plot:
                    # Show boxes
                    plt.show()

                # Create ground truth output arrays
                # x2 to include valid / invalid encoding for selecting mini-batches
                object = np.zeros((self.out_rows, self.out_cols, self.num_anchors*2), dtype="float32")
                regression = np.zeros((self.out_rows, self.out_cols, self.num_anchors*4*2), dtype="float32")

                # Set values in ground truth arrays
                for key in best_anchor_for_gt:
                    row, col = key
                    if best_anchor_for_gt[key]["type"] == "pos":
                        anchors = [ x for x in best_anchor_for_gt[key] if isinstance(x, int)]
                        for a in anchors:
                            # Set as foreground
                            object[row, col, a] = 1
                            # Set as valid
                            object[row, col, self.num_anchors + a] = 1
                            # Set regression values
                            regression[row, col, a*4:(a*4)+4] = best_anchor_for_gt[key][a]["offset"]
                            # Set as valid
                            regression[row, col, self.num_anchors + a*4:self.num_anchors + (a * 4) + 4] = [1.0]*4

                    elif best_anchor_for_gt[key]["type"] == "neutral":
                        # Set as foreground
                        object[row, col, :self.num_anchors] = [1.]*self.num_anchors
                        # Set as invalid
                        object[row, col, self.num_anchors:] = 0
                        # Set regression values as invalid
                        regression[row, col, self.num_anchors*4:] = [0.0]*self.num_anchors*4
                        pass

                    elif best_anchor_for_gt[key]["type"] == "neg":
                        # Set as valid
                        object[row, col, self.num_anchors:] = [1.0]*self.num_anchors
                        # Set as valid
                        regression[row, col, self.num_anchors * 4:] = [1.0] * self.num_anchors * 4

                # The RPN has more negative than positive regions, so we want to invalidate the majority
                # of the valid background classes so that problem is less unbalanced. It is common to
                # limit the total number to 256 anchors in total.
                num_regions = 32


                # Rescale image
                image /= 255.

                # Add to containers
                objects.append(object)
                regressions.append(regression)
                images.append(image)

            # Stack arrays on first dimension to create batch
            batch_object = np.stack(objects)
            batch_regression = np.stack(regressions)
            batch_image = np.stack(images)

            return np.copy(batch_image), np.copy(batch_object), np.copy(batch_regression)

    def __next__(self):
        return self.create_batch()

    def transform_small_coords(self, row, col):
        """
        Transforms a point the small coord space to the large
        coord space
        """
        x = col * (self.im_cols // self.out_cols)
        y = row * (self.im_rows // self.out_rows)
        return x, y

# Intersection Over Union functions
def union(au, bu, area_intersection):
    """
    a and b should be (x1,y1,x2,y2)
    """
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    """
    a and b should be (x1,y1,x2,y2)
    """
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    """
    a and b should be (x1,y1,x2,y2)
    """
    #
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def non_max_supression(bounding_boxes, overlap_threshold):
    """
    A simple implementation of non-max-supression as described by
    Andrew Ng at https://www.youtube.com/watch?v=mlswVd_IDOE

    Algorithm:
        outputs = [p, x, y, w, h]
        remove all where p <= 0.6
        while there are any remaining boxes:
            - pick the box with the largest p - output that as prediction
            - discard any remaing box with IoU >= overlap_threshold with
            the box output in the previous step

            repeat for another max p until there are no more to suppress

    """


def apply_offest(x, y, width, height, offsets):
    """
    Apply offsets and return new values of x, y width, height

    Note: offsets are  ∆ x−centre , ∆ y−centre , ∆ width , ∆ height
    """
    delta_x = offsets[0]
    delta_y = offsets[1]
    delta_w = offsets[2]
    delta_h = offsets[3]

    # Convert to centre coords
    xc = x + (width / 2)
    yc = y + (height / 2)

    # Offset
    x_off = xc + delta_x
    y_off = yc + delta_y
    width = width + delta_w
    height = height + delta_h

    # Convert to corner coords
    x = x_off - (width / 2)
    y = y_off - (height / 2)

    return x, y, width, height

# ----------- LOSS FUNCTIONS ------------ #
# A nice explanation is available at
# https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):

        # x is the difference between true value and predicted vaue
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred

        # absolute value of x
        x_abs = K.abs(x)

        # If x_abs <= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):

            return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num



# --------------------------------- MAIN ---------------------------------------- #
if __name__ == "__main__":

    annotation_dir = "/home/simon/PycharmProjects/ObjectDetection/data/annotations/"
    image_dir = "/home/simon/PycharmProjects/ObjectDetection/data/images/"

    data_gen = DataGenerator(annotation_dir, image_dir)
    data_gen.plot = False

    X, Y_object, Y_regression = next(data_gen)

    print(X.shape, Y_object.shape, Y_regression.shape)







