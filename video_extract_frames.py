import numpy as np
import glob
import os
import argparse
import time
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import time_synchronized


def init(opt):
    weights, imgsz = opt.weights, opt.img_size
    half = True if 'cuda' in opt.device else False  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=opt.device)  # load FP32 model
    if half:
        model.half()  # to FP16
    # Run inference
    if 'cuda' in opt.device:
        model(torch.zeros(1, 3, imgsz, imgsz).to(opt.device).type_as(next(model.parameters())))  # run once
    return model


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect(frame, model, opt):
    # Convert
    frame = letterbox(frame, opt.img_size)[0]
    frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    frame = np.ascontiguousarray(frame)
    
    img = torch.from_numpy(frame).to(opt.device)
    img = img.half() if 'cuda' in opt.device else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        # img = img.permute(0, 3, 1, 2)
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # Process detections
    label = dict()
    for i, det in enumerate(pred):
        if len(det):
            # for c in det[:, -1].unique():
            #     print('x', names[int(c)])
            for *_, conf, cls in reversed(det):
                conf, idx = conf.item(), int(cls)
                # print(names[idx], conf)
                if names[idx] not in label.keys():
                    label[names[idx]] = conf
                else:
                    label[names[idx]] = max(conf, label[names[idx]])
    return label


def extract(opt):
    start = time.time()
    video_paths = glob.glob(opt.src_path + '*' + opt.video_format)
    frame_counter = 0
    frame_name_counter = opt.frame_start_counter
    # model init
    model = init(opt)
    for idx, v_p in enumerate(video_paths):
        if opt.start_file != 'None.none' and os.path.split(v_p) != opt.start_file:
            continue
        video = cv2.VideoCapture(v_p)
        if not video.isOpened():
            print(f'[{idx + 1}/{len(video_paths)}] video corrupted!')
            continue
        else:
            w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = video.get(cv2.CAP_PROP_FPS)
            fc = video.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f'[{idx + 1}/{len(video_paths)}] video info, name: {os.path.split(v_p)[1]}, w: {w}, h: {h}, fps: {fps:.2f}, fc: {fc}, Time(min): {fc/fps/60:.2f}')
            success, frame = video.read()
            while success:
                if frame_counter % opt.frame_inter == 0:
                    dst_frame_path = opt.dst_path + os.path.split(v_p)[1].split('.')[0]
                    dst_frame_file_name = str(frame_name_counter) + '.jpg'
                    if not os.path.exists(dst_frame_path):
                        os.mkdir(dst_frame_path)
                    if opt.use_yolov5:
                        t1 = time_synchronized()
                        label = detect(frame, model, opt)
                        t2 = time_synchronized()
                        FLAG = True
                        for cls in opt.classes:
                            if cls not in label.keys():
                                FLAG = False
                        if not FLAG:
                            print(f'[{idx + 1}/{len(video_paths)}] no {", ".join(opt.classes)}! video: {os.path.split(v_p)[1]}, Time(min): [{frame_counter / fps / 60:.2f}/{fc / fps / 60:.2f}]')
                        else:
                            ret = cv2.imwrite(dst_frame_path + '/' + dst_frame_file_name, frame)
                            if not ret:
                                cv2.imencode('.jpg', frame)[1].tofile(dst_frame_path + '/' + dst_frame_file_name)
                            print(f'[{idx + 1}/{len(video_paths)}] extracting, video: {os.path.split(v_p)[1]}, infer time(s): {t2 - t1:.2f}, Time(min): [{frame_counter/fps/60:.2f}/{fc/fps/60:.2f}], frame: {frame_name_counter}.jpg')
                            frame_name_counter += 1
                    else:
                        cv2.imwrite(dst_frame_path + '/' + dst_frame_file_name, frame)
                        print(f'[{idx + 1}/{len(video_paths)}] extracting, video: {os.path.split(v_p)[1]}, Time(min): [{frame_counter/fps/60:.2f}/{fc/fps/60:.2f}], frame: {frame_name_counter}.jpg')
                        frame_name_counter += 1
                frame_counter += 1
                # print(frame_counter)
                success, frame = video.read()
        frame_counter = 0
    end = time.time()
    print(f"Extracting Complete! Time: {(end - start)/60:.2f} min")


# python video_extract_frames.py --src_path ./suit_hat/ --video_format .mp4 --dst_path ./suit_hat_0427/ --frame_inter 30 --use-yolov5 --classes person --device cpu
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./sanitation_suit/', help='source path of videos')
    parser.add_argument('--video_format', type=str, default='.dav', help='format of videos')
    parser.add_argument('--dst_path', type=str, default='./sanitation_suit_extract/', help='dest path of videos')
    parser.add_argument('--start_file', type=str, default='None.none', help='start video file name')
    parser.add_argument('--frame_inter', type=int, default=60, help='extract frames per frame_inter')
    parser.add_argument('--frame_start_counter', type=int, default=0, help='counter value of first frame')
    parser.add_argument('--use-yolov5', action='store_true', help='use yolov5 to filter frames')
    parser.add_argument('--classes', nargs='+', type=str, help='COCO 80 classes')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    if not os.path.exists(opt.dst_path):
        os.mkdir(opt.dst_path)
    print(opt)
    extract(opt)
