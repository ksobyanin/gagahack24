import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import json

import matplotlib.pyplot as plt

from ultralytics import YOLO

import re
import string
import json
class OCR:

    def __init__(self, model_text_bbox_recognizer, model_classification):
        self.model = YOLO(model_text_bbox_recognizer) # load a pretrained YOLOv8n model: best_text_boxes.pt
        self.model_class = YOLO(model_classification)
        self.ocr = easyocr.Reader(['en', 'ru'], gpu=True)

        self.img_path = None
        self.img = None
        self.cropped_images = None

        self.doc_classes = [
            'driver_license_back',
            'driver_license_front',
            'personal_passport_front',
            'personal_passport_registration',
            'vehicle_passport',
            'vehicle_certificate_owner',
            'vehicle_certificate_properties'
        ]

        self.doc_classes_to_type = {
            'driver_license_back': 'driver_licence',
            'driver_license_front': 'driver_licence',
            'personal_passport_front': 'personal_passport',
            'personal_passport_registration': 'personal_passport',
            'vehicle_passport': 'vehicle_passport',
            'vehicle_certificate_owner': 'vehicle_certificate',
            'vehicle_certificate_properties': 'vehicle_certificate'
        }

        self.doc_classes_to_page = {
            'driver_license_back': 2,
            'driver_license_front': 1,
            'personal_passport_front': 1,
            'personal_passport_registration': 2,
            'vehicle_passport': 0,
            'vehicle_certificate_owner': 2,
            'vehicle_certificate_properties': 1
        }

        self.doc_regex = {
            'driver_license_back': r'\d{2}\s\d{2}\s\d{6}',
            'driver_license_front': r'\d{2}\s\d{2}\s\d{6}',
            'personal_passport_front': r'\d{2}\s\d{2}\s\d{6}',
            'personal_passport_registration': None,
            'vehicle_passport': r'\d{2}\s[A-ZА-Я]{2}\s\d{6}',
            'vehicle_certificate_owner': r'\d{2}\s[A-Z]{1}\s\d{6}',
            'vehicle_certificate_properties': r'\d{2}\s[A-ZА-Я]{2}\s\d{6}'
        }

    def read_image(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(self.img_path)

    def recognize_text(self, doc_class=None):

        # print(self.cropped_images)

        flipped_cropped_images = [cv2.rotate(img_crop,
                                             cv2.ROTATE_180) for img_crop in self.cropped_images]

        # print(self.cropped_images)
        t_ = [sorted(self.ocr.readtext(img_crop),
                     key=lambda x: x[0][0][0]) for img_crop in self.cropped_images]

        t_flipped = [sorted(self.ocr.readtext(img_crop),
                     key=lambda x: x[0][0][0]) for img_crop in  flipped_cropped_images]

        texts = [self.remove_punct(' '.join([t[i][1] for i in range(len(t))])) for t in t_]
        # print(texts)
        confidences = [np.mean([t[i][2] for i in range(len(t))]) for t in t_]

        texts_flipped = [self.remove_punct(' '.join([t[i][1] for i in range(len(t))])) for t in t_flipped]
        confidences_flipped = [np.mean([t[i][2] for i in range(len(t))]) for t in t_flipped]

        if doc_class and self.doc_regex[doc_class]:
            for i, text in enumerate(texts):
                if re.findall(self.doc_regex[doc_class], text):
                    confidences[i] = 1
            for i, text in enumerate(texts_flipped):
                if re.findall(self.doc_regex[doc_class], text):
                    confidences_flipped[i] = 1

        avg_conf = np.mean(confidences)
        avg_conf_flipped = np.mean(confidences_flipped)

        if avg_conf_flipped > avg_conf:
            texts = texts_flipped
            confidences = confidences_flipped

        return texts, confidences

    def plot_recognized_boxes(self):
        # read image
        img = cv2.imread(self.img_path)

        im_h = img.shape[0]
        im_w = img.shape[1]

        recognized_bboxes = self.get_recognized_bboxes()
        # print(recognized_bboxes)

        for conv_bbox in recognized_bboxes:

            self.x_center = int(conv_bbox[0])
            self.y_center = int(conv_bbox[1])
            self.width = int(conv_bbox[2])
            self.height = int(conv_bbox[3])

            x_1 = int(self.x_center - self.width / 2)
            y_1 = int(self.y_center - self.height / 2)
            x_2 = int(self.x_center + self.width / 2)
            y_2 = int(self.y_center + self.height / 2)

            cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (0, 255, 0), 5)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def crop_image_to_bboxes(self, show_img=False):

        self.cropped_images = []

        bboxes_list = self.get_recognized_bboxes()

        # print(bboxes_list)

        for bbox in bboxes_list:

            x_center, y_center, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

            x_1, y_1, x_2, y_2 = bbox[4], bbox[5], bbox[6], bbox[7]

            img_crop = self.img[y_1:y_2, x_1:x_2]

            img_crop_height, img_crop_width, _ = img_crop.shape

            # rotate if width < height
            if img_crop_width < img_crop_height:
                img_crop = cv2.rotate(img_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

            self.cropped_images.append(img_crop)

            if show_img:
                plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
                plt.show()

        return self.cropped_images

    def get_recognized_bboxes(self):

        model_output = self.model(self.img_path, verbose=False)

        result = model_output[0].boxes.xywh.tolist()
        # print(result)

        bboxes_list = []

        for bbox in result:

            # print(bbox)

            x_center, y_center, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

            x_1 = int(x_center - width / 2)
            y_1 = int(y_center - height / 2)
            x_2 = int(x_center + width / 2)
            y_2 = int(y_center + height / 2)

            bboxes_list.append([x_center, y_center, width, height, x_1, y_1, x_2, y_2])

        return bboxes_list

    def remove_punct(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def predict_class(self):

        result = self.model_class(self.img_path, verbose=False)

        cls = result[0].boxes.cls.tolist()
        conf = result[0].boxes.conf.tolist()

        return self.doc_classes[int(cls[np.argmax(conf)])], conf

    def predict_and_return_json(self, img_or_path, show_plot=False):
        if isinstance(img_or_path , (str, Path)):
            self.read_image(img_or_path)
        else:
            self.image = cv2.cvtColor(img_or_path, cv2.COLOR_RGB2BGR)

        doc_class, conf = self.predict_class()

        self.get_recognized_bboxes()
        self.crop_image_to_bboxes(show_plot)

        if doc_class != self.doc_classes[3]:
            texts, text_confs = self.recognize_text(doc_class)
            text = texts[np.argmax(text_confs)]
            if len(text.split()) > 2:
                if ' J' in text:
                    text = text.replace(' J', '')
                series = text.split()[0] + text.split()[1]
                number = text.split()[2]
            else:
                series = None
                number = text

        else:
            series = None
            number = None

        return json.dumps({
                'type': self.doc_classes_to_type[doc_class],
                'confidence': conf[0],
                'series': series,
                'number': number,
                'page_number': self.doc_classes_to_page[doc_class]
            })
    

