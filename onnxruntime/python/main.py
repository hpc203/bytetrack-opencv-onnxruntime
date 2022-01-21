import os
import copy
import time
import argparse
import cv2
from loguru import logger
from byte_tracker.byte_tracker_onnx import ByteTrackerONNX

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

    byte_tracker = ByteTrackerONNX(args)

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
            cv2.namedWindow('ByteTrack ONNX Sample', 0)
            cv2.imshow('ByteTrack ONNX Sample', debug_image)
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
