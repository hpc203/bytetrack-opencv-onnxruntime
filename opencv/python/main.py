import os
import copy
import time
import argparse
import cv2
from loguru import logger
import numpy as np
from byte_tracker.utils.yolox_utils import (
    post_process,
    multiclass_nms,
)
from byte_tracker.tracker.byte_tracker import BYTETracker

def pre_process(image, input_size, mean, std):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

class ByteTracker(object):
    def __init__(self, args):
        self.args = args

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.net = cv2.dnn.readNet(args.model)
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

        blob = cv2.dnn.blobFromImage(image)
        self.net.setInput(blob)
        result = self.net.forward()

        dets = self._post_process(result, image_info)

        bboxes, ids, scores = self._tracker_update(
            dets,
            image_info,
        )

        return image_info, bboxes, ids, scores

    def _post_process(self, result, image_info):
        predictions = post_process(
            result,
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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--use_debug_window',
        action='store_true',
    )

    parser.add_argument(
        '--model',
        type=str,
        default='byte_tracker/model/bytetrack_s.onnx',
    )
    parser.add_argument(
        '--video',
        type=str,
        default='sample.mp4',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.7,
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default='608,1088',
    )
    parser.add_argument(
        '--with_p6',
        action='store_true',
        help='Whether your model uses p6 in FPN/PAN.',
    )

    # tracking args
    parser.add_argument(
        '--track_thresh',
        type=float,
        default=0.5,
        help='tracking confidence threshold',
    )
    parser.add_argument(
        '--track_buffer',
        type=int,
        default=30,
        help='the frames for keep lost tracks',
    )
    parser.add_argument(
        '--match_thresh',
        type=float,
        default=0.8,
        help='matching threshold for tracking',
    )
    parser.add_argument(
        '--min-box-area',
        type=float,
        default=10,
        help='filter out tiny boxes',
    )
    parser.add_argument(
        '--mot20',
        dest='mot20',
        default=False,
        action='store_true',
        help='test mot20.',
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    use_debug_window = args.use_debug_window

    video_path = args.video
    output_dir = args.output_dir

    byte_tracker = ByteTracker(args)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not use_debug_window:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, video_path.split("/")[-1])
        logger.info(f"video save path is {save_path}")

        video_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(width), int(height)),
        )

    frame_id = 1
    winName = 'Deep learning object detection in OpenCV'
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        _, bboxes, ids, scores = byte_tracker.inference(frame)

        elapsed_time = time.time() - start_time

        debug_image = draw_tracking_info(
            debug_image,
            bboxes,
            ids,
            scores,
            frame_id,
            elapsed_time,
        )

        if use_debug_window:
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            cv2.namedWindow(winName, 0)
            cv2.imshow(winName, debug_image)
        else:
            video_writer.write(debug_image)

        logger.info(
            'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                             elapsed_time * 1000), )
        frame_id += 1

    if use_debug_window:
        cap.release()
        cv2.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_tracking_info(
    image,
    tlwhs,
    ids,
    scores,
    frame_id=0,
    elapsed_time=0.,
):
    text_scale = 1.5
    text_thickness = 2
    line_thickness = 2

    text = 'frame: %d ' % (frame_id)
    text += 'elapsed time: %.0fms ' % (elapsed_time * 1000)
    text += 'num: %d' % (len(tlwhs))
    cv2.putText(
        image,
        text,
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 255, 0),
        thickness=text_thickness,
    )

    for index, tlwh in enumerate(tlwhs):
        x1, y1 = int(tlwh[0]), int(tlwh[1])
        x2, y2 = x1 + int(tlwh[2]), y1 + int(tlwh[3])

        color = get_id_color(ids[index])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # text = str(ids[index]) + ':%.2f' % (scores[index])
        text = str(ids[index])
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 0, 0), text_thickness + 3)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (255, 255, 255), text_thickness)
    return image


if __name__ == '__main__':
    main()
