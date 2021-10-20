import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os
from datetime import datetime


class LiveVideoWindow(QWidget):
    def __init__(self, window1=None):
        super(LiveVideoWindow, self).__init__()
        self.setWindowTitle('PPE Detection')
        self._window1 = window1
        self.VBL = QVBoxLayout()

        self.label = QLabel()
        self.label.setText("Team MA_A_20")
        self.label.setAlignment(Qt.AlignLeft)
        self.VBL.addWidget(self.label)

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        # stop video button
        self.CancelBTN = QPushButton("Stop Video Feed")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        # start video button
        self.StartBTN = QPushButton("Start Video Feed")
        self.StartBTN.clicked.connect(self.StartFeed)
        self.VBL.addWidget(self.StartBTN)
        self.StartBTN.setDisabled(True)

        # save frame button
        self.SaveBTN = QPushButton("Save Current Frame")
        self.SaveBTN.clicked.connect(self.SaveFrame)
        self.VBL.addWidget(self.SaveBTN)

        # reports button
        self.ReviewBTN = QPushButton("Review Reports")
        self.ReviewBTN.clicked.connect(self.ReviewReportsButton)
        self.VBL.addWidget(self.ReviewBTN)

        # centres application on screen
        qtRectangle = self.frameGeometry()  # get window dimensions
        centerPoint = QDesktopWidget().availableGeometry().center()  # get coordinates of centre of screen
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())  # move window to centre of screen

        self.Worker1 = Worker1()

        # start process for receiving live video
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)

    # slot to receive signal
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))  # updates the latest frame to be shown

    def CancelFeed(self):
        # stops receiving video feed
        self.Worker1.stop_video()
        self.StartBTN.setDisabled(False)
        self.CancelBTN.setDisabled(True)
        self.SaveBTN.setDisabled(True)

    def StartFeed(self):
        # starts receiving video feed
        self.Worker1.ThreadActive = True
        self.CancelBTN.setDisabled(False)
        self.StartBTN.setDisabled(True)
        self.SaveBTN.setDisabled(False)

    def SaveFrame(self):
        # saves current frame
        self.Worker1.save_frame()

    def ReviewReportsButton(self):
        # open saved frames folder
        user = os.getenv('username')
        path = "C:/Users/" + user + "/Desktop/Reports"
        path = os.path.realpath(path)
        os.startfile(path)


class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)    # signal to emit back to main window

    def __init__(self):
        super().__init__()
        self.ThreadActive = True
        self.frame = None

    def run(self):
        save_img = False
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()

        while self.ThreadActive:

            for path, img, im0s, vid_cap in dataset:
                save = False
                detections = [0, 0, 0, 0, 0]  # array for number of detections for each ppe
                person_array = [0]

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            occ = int(f"{n}")
                            if names[int(c)] == "Person":
                                person_array[0] = occ
                            elif names[int(c)] == "Helmet":
                                detections[0] = occ
                            elif names[int(c)] == "Mask":
                                detections[1] = occ
                            elif names[int(c)] == "Vest":
                                detections[2] = occ
                            elif names[int(c)] == "Glove":
                                detections[3] = occ
                            elif names[int(c)] == "Goggles":
                                detections[4] = occ

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    # Print time (inference + NMS)
                    # print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Stream results
                    if view_img:
                        if self.ThreadActive:
                            self.frame = im0
                            Image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                            # FlippedImage = cv2.flip(Image, 1)
                            ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                            Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                            self.ImageUpdate.emit(Pic)  # emits the frame to be shown on the main window

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video'
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fourcc = 'mp4v'  # output video codec
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                            vid_writer.write(im0)

                    # if 0, means no person in frame
                    if person_array[0] != 0:
                        # check if each number of ppe appeared and number of person is the same, if different
                        # means a worker did not wear all 5 ppe
                        flag = True
                        for j in range(len(detections)):
                            if person_array[0] != detections[j]:
                                flag = False

                        if not flag:
                            save = True
                        else:
                            save = False
                    else:
                        save = False

                    # save frames without 5 ppe
                    if save and self.ThreadActive:
                        user = os.getenv('username')
                        path = "C:/Users/" + user + "/Desktop/Reports/Auto Save"
                        current = datetime.now()
                        dt = current.strftime("%Y%m%d_%H%M%S")
                        filename = path + "/img" + "_" + dt + ".jpg"
                        cv2.imwrite(filename, im0)

            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                print(f"Results saved to {save_dir}{s}")

            # print(f'Done. ({time.time() - t0:.3f}s)')

    def stop_video(self):
        self.ThreadActive = False

    def save_frame(self):
        # saves current frame to destination
        user = os.getenv('username')
        path = "C:/Users/" + user + "/Desktop/Reports/Manual Save"
        current = datetime.now()
        dt = current.strftime("%Y%m%d_%H%M%S")
        filename = path + "/img" + "_" + dt + ".jpg"
        cv2.imwrite(filename, self.frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    # check_requirements()

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()

    # make directory at desktop
    user = os.getenv('username')
    path = "C:/Users/" + user + "/Desktop/Reports"
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    path2 = "C:/Users/" + user + "/Desktop/Reports/Auto Save"
    try:
        os.mkdir(path2)
    except OSError as error:
        print(error)

    path3 = "C:/Users/" + user + "/Desktop/Reports/Manual Save"
    try:
        os.mkdir(path3)
    except OSError as error:
        print(error)

    App = QApplication(sys.argv)
    Root = LiveVideoWindow()
    Root.show()
    sys.exit(App.exec())
