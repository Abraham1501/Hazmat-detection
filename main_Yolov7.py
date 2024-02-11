from pathlib import Path
import datetime

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel


class Yolo_v7:
    def __init__(self, save_images, save_txt, print_results):
        # inicial parameters
        self.opt_weigths = 'best.pt'  # model.pt path(s)
        self.opt_img_size = 640       # inference size (pixels)
        self.opt_conf_thres = 0.45    # object confidence threshold
        self.opt_iou_thres = 0.25     # IOU threshold for NMS

        self.opt_classes = None       # filter by class: --class 0, or --class 0 2 3
        self.opt_agnostic_nms = True  # class-agnostic NMS
        self.opt_save_conf = False    # save confidences in --save-txt labels
        self.opt_no_trace = True      # don`t trace model
        self.opt_augment = True       # augmented inference
        self.opt_device = 'cpu'       # cuda device, i.e. 0 or 0,1,2,3 or cpu

        self.opt_print_results = print_results
        self.opt_save_img_results = save_images #saves images
        self.opt_save_txt_results = save_txt    #saves detections in .txt

        #model
        self.model = None
        self.half = None
        self.stride = None
        self.device = None
        self.names = None
        self.colors = None
        self.old_img_w = self.old_img_h = self.old_img_b = None

        #Directories
        self.save_dir = Path('_results')
        self.txt_path = self.save_dir / 'txt'
        self.img_path = self.save_dir / 'images'


    def inicialize(self):
        #Initialize
        set_logging()
        device = select_device(self.opt_device)
        half = device.type != 'cpu' #half precision olny supported on CUDA

        #Load model
        model = attempt_load(self.opt_weigths, map_location=device) #load FP32 model
        stride = int(model.stride.max()) #model stride
        self.opt_img_size = check_img_size(self.opt_img_size, s=stride)  # check img_size

        if not self.opt_no_trace:
            model = TracedModel(model, device, self.opt_img_size)
        if half:
            model.half() # to FP16

        #Set Dataloader
        cudnn.benchmark = True # set True to speed up constant image size inference

        #Get names and colors
        names = model.module.mames if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        #Run inference
        if device.type != 'cpu':
            model(torch.zeros(1,3, self.opt_img_size, self.opt_img_size).to(device).type_as(next(model.parameters()))) #run once

        self.old_img_w = self.old_img_h = self.opt_img_size
        self.old_img_b = 1

        self.model = model
        self.half = half
        self.stride = stride
        self.device = device
        self.names = names
        self.colors = colors

    def detect(self, img0, save):

        # Padded resize
        img = letterbox(img0, self.opt_img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() #uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        #Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.opt_augment)[0]

        #Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.opt_augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt_conf_thres, self.opt_iou_thres, classes=self.opt_classes,
                                   agnostic=self.opt_agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                if self.opt_print_results or self.opt_save_txt_results:
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)


            # Save results
            if self.opt_save_img_results and save: # image with detections
                count = 0
                while (self.img_path / ('image_' + str(count) + '.jpg')).exists():  # img.jpg
                    count += 1
                img_path = str(self.img_path / ('image_' + str(count) + '.jpg'))

                cv2.imwrite(img_path, img0)
                print(f" The image with the result is saved in: {img_path}")

            if self.opt_save_txt_results: # write results
                with open(str(self.txt_path / 'detections.txt'), 'a') as f:
                    f.write(str(datetime.datetime.now()) + s + '\n')

                if save:
                    count = 0
                    while (self.txt_path / ('write_' + str(count) + '.txt')).exists():
                        count += 1
                    txt_path = str(self.txt_path / ('write_' + str(count) + '.txt'))
                    print(txt_path)

                    with open(txt_path, 'a') as f:
                        f.write(s + '\n')

            # Print results
            if self.opt_print_results:
                print(f"Detections:  {s}")

        return img0