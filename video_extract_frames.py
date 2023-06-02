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
from utils import get_file_name, get_file_prefix


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
    video_paths = glob.glob(os.path.join(opt.video_folder_path, '*' + opt.video_format))
    if opt.filter:
        model = init(opt)
        print(f"[INFO] yolov5 model init, Done.")
    for idx, v_p in enumerate(video_paths):
        v_p_name, v_p_prefix = get_file_name(v_p), get_file_prefix(v_p)
        frame_counter, counter = 0, 0
        if opt.first_video and v_p_name != opt.first_video:
            continue
        opt.first_video = False
        video = cv2.VideoCapture(v_p)
        if not video.isOpened():
            print(f"[WARNING] {idx + 1}/{len(video_paths)} video corrupted!")
            continue
        else:
            w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = video.get(cv2.CAP_PROP_FPS)
            fc = video.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"[INFO] {idx + 1}/{len(video_paths)} video: {v_p_name}, W: {w}, H: {h}, FPS: {fps:.2f}, FC: {fc}, Time(min): {fc / fps / 60:.2f}.")
            success, frame = video.read()
            while success:
                if frame_counter % opt.frame_inter == 0:
                    dst_frame_path = os.path.join(opt.save_dir, v_p_prefix)
                    dst_frame_file_name = v_p_prefix + '_' + str(counter) + opt.save_format
                    if not os.path.exists(dst_frame_path):
                        os.mkdir(dst_frame_path)
                        print(f"[INFO] Creating save_dir: {dst_frame_path}, Done.")
                    if opt.filter:
                        t1 = time_synchronized()
                        label = detect(frame, model, opt)
                        t2 = time_synchronized()
                        if opt.relationship == 'and':
                            FLAG = True
                            for cls in opt.classes:
                                if cls not in label.keys():
                                    FLAG = False
                        if opt.relationship == 'or':
                            FLAG = False
                            for cls in opt.classes:
                                if cls in label.keys():
                                    FLAG = True
                        if not FLAG:
                            print(f"[INFO] Frame at Time(s): {frame_counter / fps:.2f}/{fc / fps:.2f} of {idx + 1}/{len(video_paths)} video: {v_p_name} filtered!")
                        else:
                            ret = cv2.imwrite(os.path.join(dst_frame_path, dst_frame_file_name), frame)
                            if not ret:
                                # for chinese
                                cv2.imencode(opt.save_format, frame)[1].tofile(os.path.join(dst_frame_path, dst_frame_file_name))
                            counter += 1
                            print(f"[INFO] With inference time(s): {t2 - t1:.2f}, Frame at Time(s): {frame_counter / fps:.2f}/{fc / fps:.2f} of {idx + 1}/{len(video_paths)} video: {v_p_name} saved with {dst_frame_file_name}!")
                    else:
                        cv2.imwrite(os.path.join(dst_frame_path, dst_frame_file_name), frame)
                        counter += 1
                        print(f"[INFO] Frame at Time(s): {frame_counter / fps:.2f}/{fc / fps:.2f} of {idx + 1}/{len(video_paths)} video: {v_p_name} saved with {dst_frame_file_name}!")
                frame_counter += 1
                success, frame = video.read()
    end = time.time()
    print(f"[INFO] Frames extracting done in {(end - start) / 60:.2f} min!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder_path', type=str, default=None, required=True, help='folder path of videos.')
    parser.add_argument('--video_format', type=str, default='.mp4', help='format of videos.')
    parser.add_argument('--save_dir', type=str, default=None, required=True, help='save path of images extracting from videos.')
    parser.add_argument('--save_format', type=str, default='.jpg', help='format of saved images.')
    parser.add_argument('--first_video', type=str, default=None, help='first video file before extracting.')
    parser.add_argument('--frame_inter', type=int, default=60, help='frame interval.')
    parser.add_argument('--filter', action='store_true', help='use yolov5 to filter frames.')
    parser.add_argument('--classes', default=None, nargs='+', type=str, help='classes of interest.')
    parser.add_argument('--relationship', default='and', type=str, help='the relationship of classes of interest, and/or.')
    parser.add_argument('--weights', type=str, default='./yolov5m.pt', help='weight path of yolov5 model.')
    parser.add_argument('--img-size', type=int, default=640, help='inference size of images.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS.')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print(f"[INFO] Options: {opt}")
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
        print(f"[INFO] Creating save_dir: {opt.save_dir}, Done.")
    assert opt.filter == bool(opt.classes), f"[ERROR] If you want to filter frames, please provide --filter and --classes both."
    assert opt.relationship in ['and', 'or'], f"[ERROR] Pleas choose right relationship in frame: and/or"
    extract(opt)
