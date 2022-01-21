#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import numpy as np
import onnxruntime

from byte_tracker.utils.yolox_utils import (
    pre_process,
    post_process,
    multiclass_nms,
)
from byte_tracker.tracker.byte_tracker import BYTETracker


class ByteTrackerONNX(object):
    def __init__(self, args):
        self.args = args

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.session = onnxruntime.InferenceSession(args.model)
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

        self.tracker = BYTETracker(args, frame_rate=30)

    def _pre_process(self, image):
        image_info = {'id': 0}

        image_info['image'] = copy.deepcopy(image)
        image_info['width'] = image.shape[1]
        image_info['height'] = image.shape[0]

        preprocessed_image, ratio = pre_process(
            image,
            self.input_shape,
            self.rgb_means,
            self.std,
        )
        image_info['ratio'] = ratio

        return preprocessed_image, image_info

    def inference(self, image):
        image, image_info = self._pre_process(image)

        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: image[None, :, :, :]})

        dets = self._post_process(result, image_info)

        bboxes, ids, scores = self._tracker_update(
            dets,
            image_info,
        )

        return image_info, bboxes, ids, scores

    def _post_process(self, result, image_info):
        predictions = post_process(
            result[0],
            self.input_shape,
            p6=self.args.with_p6,
        )
        predictions = predictions[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= image_info['ratio']

        dets = multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.args.nms_th,
            score_thr=self.args.score_th,
        )

        return dets

    def _tracker_update(self, dets, image_info):
        online_targets = []
        if dets is not None:
            online_targets = self.tracker.update(
                dets[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        return online_tlwhs, online_ids, online_scores
