# -*- coding: utf-8 -*-
"""
Created on: 4/1/2025
Original author: Adil Zaheer
"""
# Built-Ins
import logging
from collections import Counter

# Third Party
# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import tensorflow as tf
import torch

LOG = logging.getLogger(__name__)


def iou_validation(
    box_predictions: tf.Tensor, box_truth: tf.Tensor, box_format: str, model_package: str
):
    """
    Intersection over Union to validate image segmentation. Able to process
    YOLO/COCO (0.1, 0.5, 0.4 etc.) and PASCAL VOC format (100, 300, 200 etc.)

    param: box_predictions: Tensor object. Contains the predicted bounding box
                            values. These are either in (x1,y1,w,h) format
                            for YOLO/COCO models or (x1,y1,x2,y2) format for
                            PASCAL VOC.
    param: box_truth: Tensor object. These are the truth bounding box values.
                      These must be in the same format as box_predictions.
    param: box_format: String. Either midpoint, centre or None. Midpoint would
                       mean (x1,y1,w,h) so YOLO/COCO and centre would mean
                       (x1,y1,x2,y2) so PASCAL VOC. If unsure, leave as None
                       and the function will attempt to work out the format.
    param: model_package: String. Either TensorFlow or PyTorch. If None then
                          Tensorflow is used as default.

    returns: Intersection over Union for an image.
    """
    intersection = None

    def is_yolo_coord(box_predictions):
        return all(0 <= value <= 1 for value in box_predictions)

    if is_yolo_coord(box_predictions) or box_format == "midpoint":
        box1_x1 = box_predictions[..., 0:1] - box_predictions[..., 2:3] / 2
        box1_y1 = box_predictions[..., 1:2] - box_predictions[..., 3:4] / 2
        box1_x2 = box_predictions[..., 0:1] - box_predictions[..., 2:3] / 2
        box1_y2 = box_predictions[..., 1:2] - box_predictions[..., 3:4] / 2
        box2_x1 = box_truth[..., 0:1] - box_truth[..., 2:3] / 2
        box2_y1 = box_truth[..., 1:2] - box_truth[..., 3:4] / 2
        box2_x2 = box_truth[..., 0:1] - box_truth[..., 2:3] / 2
        box2_y2 = box_truth[..., 1:2] - box_truth[..., 3:4] / 2
    else:
        box1_x1 = box_predictions[..., 0:1]
        box1_y1 = box_predictions[..., 1:2]
        box1_x2 = box_predictions[..., 2:3]
        box1_y2 = box_predictions[..., 3:4]
        box2_x1 = box_truth[..., 0:1]
        box2_y1 = box_truth[..., 1:2]
        box2_x2 = box_truth[..., 2:3]
        box2_y2 = box_truth[..., 3:4]

    if model_package == "pytorch":
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        intersection = x2 - x1.clamp(0) * y2 - y1.clamp(0)

    if model_package == "tensorflow" or model_package is None:
        x1 = tf.reduce_max(box1_x1, box2_x1)
        y1 = tf.reduce_max(box1_y1, box2_y1)
        x2 = tf.reduce_min(box1_x2, box2_x2)
        y2 = tf.reduce_min(box1_y2, box2_y2)

        x1 = tf.clip_by_value(x1, clip_value_min=-0)
        y1 = tf.clip_by_value(y1, clip_value_min=-0)
        intersection = x2 - x1 * y2 - y1

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    if intersection is not None:
        result = intersection / (box1_area + box2_area - intersection + 1e-6)
    else:
        LOG.error(
            "Bounding box intersection is still None. Either \
                   PyTorch or TensorFlow must be used to compute \
                   intersection over union. Please evaluate results."
        )
        raise ValueError(
            "Bounding box intersection is still None. Either \
                          PyTorch or TensorFlow must be used to compute \
                          intersection over union. Please evaluate results."
        )

    return result


# todo set box_format to None by default
def nms(
    predicted_bboxes: list,
    probability_threshold: int,
    iou_threshold: int,
    box_format: str,
    model_package: str,
):
    """
    Non-max suppression bounding box evaluation. This _functions uses the
    iou_validation function.

    param: predicted_bboxes: List. This should be a list of lists that contain
                             the predicted bounding box information. This will
                             be in the format:
                             PASCAL VOC: [class, probability, x1, y1, x2, y2]
                             YOLO/COCO: [class, probability, x1, y1, W, H]
    param: probability_threshold: Integer. This is a hyperparameter
                                  input that sets a minimum threshold for
                                  bounding box prediction accuracy. If the
                                  probability of a bounding box is below
                                  the threshold, it will not be considered in
                                  the non-max suppression function.

    param: iou_threshold: Integer. This is a hyperparameter input that acts as
                          a threshold for the iou comparison between a
                          predicted bounding box and the truth bounding box.
                          See iou_validation function.
    param: box_format: String. Either midpoint, centre or None. Midpoint would
                       mean (x1,y1,w,h) so YOLO/COCO and centre would mean
                       (x1,y1,x2,y2) so PASCAL VOC. If unsure, leave as None
                       and the function will attempt to work out the format.
    param: model_package: String. Either TensorFlow or PyTorch. If None then
                          Tensorflow is used as default.

    returns: List of best predicted bounding boxes.
    """

    if type(predicted_bboxes) is not list:
        LOG.error(
            "bbox must be a list that contains the bounding box \
                   information. This can be in the form of \
                   [class, probability, x1, y1, x2, y2] for PASCAL VOC or \
                   [class, probability, x1, y1, W, H] for YOLO/COCO."
        )
        raise TypeError(
            "bbox must be a list that contains the bounding box \
                         information. This can be in the form of \
                         [class, probability, x1, y1, x2, y2] for PASCAL VOC or \
                         [class, probability, x1, y1, W, H] for YOLO/COCO."
        )

    if model_package is None:
        model_package = "tensorflow"

    bboxes_post_nms = []

    # only want bbox that is over a certain probability threshold
    bboxes = [box for box in predicted_bboxes if predicted_bboxes[1] > probability_threshold]

    # sort so the largest probability bbox is what we are working with
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    while bboxes:
        selected_bbox = bboxes.pop(0)

        # first check if box and comparison box are of the same class
        # if bbox of the same class, compare (using iou). If less than threshold, then append.
        bboxes = [
            box
            for box in bboxes
            if box[0] != selected_bbox[0]
            or iou_validation(
                box_predictions=torch.tensor(selected_bbox[2:]),
                box_truth=torch.tensor(box[2:]),
                box_format=box_format,
                model_package=model_package,
            )
            < iou_threshold
        ]

        bboxes_post_nms.append(selected_bbox)

    return bboxes_post_nms


def mean_average_precision(
    prediction_bboxes: list,
    box_truth: list,
    iou_threshold: int,
    box_format: str,
    num_classes: int,
    model_package,
):
    """
    Mean Average Precision for a single iou threshold.

    [[train_index, class_pred, prob_score, x1, y1, x2, y2], [], []]


    """
    average_precisions = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # making sure correct class prediction is being compared (e.g. class 3 compared with class 3)
        for detection in prediction_bboxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in box_truth:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # need to keep track of target bboxes covered so far
        # only first bbox that covers target is correct
        # cant have multiple pred for one bounding box and count that as correct. only one per target

        # creates dict with no. of bboxes for each image. 0:3 would be image 0 has 3 bboxes
        amount_bboxes = Counter(gt[0] for gt in ground_truths)

        # create tensors of zeros that represent no. bboxes per image
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sorting over prob scores hence x[2]
        detections.sort(key=lambda x: x[2], reverse=True)

        true_positives = torch.zeros((len(detections)))
        false_positives = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            # only taking ground truths that has same index as detected bbox. can't compare against diff training images
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            # no. target bboxes for this image
            num_gts = len(ground_truth_img)
            best_iou = 0

            # going through all the gt bboxes for this image
            for idx, gt in enumerate(ground_truth_img):
                iou = iou_validation(
                    box_predictions=torch.tensor(detection[3:]),
                    box_truth=torch.tensor(gt[3:]),
                    box_format=box_format,
                    model_package=model_package,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # at this point, we have taken a single bbox for a particular class in a particular image
            # we have taken all the ground truths for that image and compared that bbox with the target bbox
            # used iou to score and kept track of best iou

            # check if prediction is correct but also not covered before (using counter)
            if best_iou > iou_threshold:
                # training idx of current bbox prediction, get idx of best gt idx (target)
                # initialise it as zero so it's not yet been covered
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    true_positives[detection_idx] = 1
                    # set to 1 as it is now covered
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    # this else is if the bbox was already covered
                    false_positives[detection_idx] = 1
            else:
                # means iou was not over threshold so not a true positive
                false_positives[detection_idx] = 1

        true_positives_cumsum = torch.cumsum(true_positives, dim=0)
        false_positives_cumsum = torch.cumsum(false_positives, dim=0)
        recalls = true_positives_cumsum / (total_true_bboxes + 1e-6)
        precisions = torch.divide(
            true_positives_cumsum, (true_positives_cumsum + false_positives_cumsum + 1e-6)
        )

        # need to add 1 to precisions as need to start at point (0,1) for numerical integration
        # add 0 for recalls as this is x-axis. Precisions is y.
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # trapz is the trapezoidal rule (so area under the graph). (y, x) input
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
