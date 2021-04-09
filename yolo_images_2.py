import winsound

import cv2
import numpy as np
import glob

from numpy.linalg import norm

import os

class Detection:
    def __init__(self):
        # Load Yolo
        self.IsDetected=False
        self.net = cv2.dnn.readNet("C:/Users/DELL/PycharmProjects"
                                   +"/deep_learning_Custom_model/"
                                   +"yolov3_cutom_last 0_06 13300.weights",
                                   "C:/Users/DELL/PycharmProjects/"
                                   +"deep_learning_Custom_model/"
                                   +"yolov3_cutom_with_6_classes.cfg")

        # net = cv2.dnn.readNet("yolov4-obj_last.weights", "yolov4-obj.cfg")
        classes = []
        with open("C:/Users/DELL/PycharmProjects/deep_learning_Custom_model/obj.names"
                ,
                  "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()

        self.output_layers = [
            self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]

        self.colors = np.random.uniform(0, 255, size=(len(classes), 3))
        self.number_of_images=0
        # Images path
        # images_path = glob.glob(r"F:\FYP\7.datasets\2.lion\f84ecc0c-8a09-470a-a31d-81b6612ae5f1\New folder\*.jpg")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\bd9024d9-486d-4f04-a78c-b666b0f395da\*.jpg")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\combine3(10,11,12,13,14,15)\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\combine\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\images18\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\5.AK47 And Hand gun\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\4.new Guns\1.Images\combination(1,2,3,4)\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\4.new Guns\1.Images\5\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\5.With out guns\1\*.png")
        #     images_path = glob.glob(r"F:\FYP\8.gun detected images\5\Robber shoves handgun into face of Jimmy Johns employee\*.png")
        # images_path = glob.glob(r"F:\FYP\10.Redmi 8 videos\1\original\*.jpg")
        # self.images_path = glob.glob(r"F:\FYP\10.Redmi 8 videos\13_10_2020\1.original pics\1\sample and original\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\6.21_10_200\1.M_16\6.All in one\*.png")
        self.images_path = glob.glob(
            r"F:\FYP\10.Redmi 8 videos\3_9_2020\2\original images\1st half\original\*.png"
        )

        self.images_path = glob.glob(
            r"E:\4.FYP 3rd folder\3.Videos\1. 31 Mar 2021\1.Pics\*.png"
        )
#         self.images_path = glob.glob(r"F:\FYP\10.Redmi 8 videos\3_9_2020\2\original images\1st half\original\2.At 275 degree\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\2.AK_47_ 9_nov_2020\1.pic\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\1. 8_nov_2020\3\1.pics\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\3.imageProcessed\1.11_11_2020\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\3.imageProcessed\2\*.png")
        # self.images_path = glob.glob(r"E:\4.FYP 3rd folder\3.Videos\1. 31 Mar 2021\1.Pics\*.png")

    def detect_images(self,img_path):
        img1 = cv2.imread(img_path)

        img2 = cv2.imread(img_path)
        img2 = cv2.resize(img1, None, fx=0.3, fy=0.3)
        print(np.average(norm(img2, axis=2)) / np.sqrt(3))

        ##standard brightness and contrast
        # alpha=20
        # beta=40
        ##testable brightness and contrast
        alpha = 1
        beta = 80

        img = cv2.addWeighted(img1,
                              alpha,
                              np.zeros(img1.shape, img1.dtype),
                              0,
                              beta)
        print(np.average(norm(img, axis=2)) / np.sqrt(3))
        img = cv2.resize(img, None, fx=0.3, fy=0.3)

        # img = cv2.resize(img1, None, fx=0.3, fy=0.3)

        cv2.moveWindow("Image", 40, 100)
        cv2.moveWindow("Image2", 500, 100)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img,
                                     0.00392,
                                     (416, 416),
                                     (0, 0, 0),
                                     True,
                                     crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        # class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            # print(out)
            for detection in out:
                # print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # Object detected
                    # number_of_objects = number_of_objects + 1
                    # print('number_of_objects = ' + str(number_of_objects))
                    # winsound.Beep(1000, 250)  # Beep at 1000 Hz for 100 ms
                    self.IsDetected=True
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    # class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                # label = str(self.classes[class_ids[i]])
                # color = self.colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), 1, 2)
                # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        directory = r'D:\1.FYP second folder\2.data set\1. 8_nov_2020\3\1.pics\1.detected'
        # directory = r'D:\1.FYP second folder\3.Train model and their pics\3.Check models which are correct\1.Five classes\3.Models_yolov3_custom_last (6)'

        # Change the current directory
        # to specified directory
        os.chdir(directory)

        # Filename
        filename = str(self.number_of_images) + '.png'
        self.number_of_images = self.number_of_images + 1
        print(self.number_of_images)
        # Using cv2.imwrite() method

        # Saving the image
        cv2.imwrite(filename, img)
        cv2.imshow("Image", img2)
        cv2.imshow("Image2", img)

        return img

    def get_images(self):
        # Load Yolo
        self.net = cv2.dnn.readNet(
            "yolov3_cutom_last 0_06 13300.weights",
            "yolov3_cutom_with_6_classes.cfg"
        )

        # net = cv2.dnn.readNet("yolov4-obj_last.weights", "yolov4-obj.cfg")
        classes = []
        with open("obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Images path
        # images_path = glob.glob(r"F:\FYP\7.datasets\2.lion\f84ecc0c-8a09-470a-a31d-81b6612ae5f1\New folder\*.jpg")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\bd9024d9-486d-4f04-a78c-b666b0f395da\*.jpg")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\combine3(10,11,12,13,14,15)\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\combine\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\2.  3d models\images18\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\1.AK-47 gun\5.AK47 And Hand gun\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\4.new Guns\1.Images\combination(1,2,3,4)\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\4.new Guns\1.Images\5\*.png")
        # images_path = glob.glob(r"F:\FYP\7.datasets\5.With out guns\1\*.png")
        #     images_path = glob.glob(r"F:\FYP\8.gun detected images\5\Robber shoves handgun into face of Jimmy Johns employee\*.png")
        # images_path = glob.glob(r"F:\FYP\10.Redmi 8 videos\1\original\*.jpg")
 #        self.images_path = glob.glob(
 # r"F:\FYP\10.Redmi 8 videos\13_10_2020\1.original pics\1\sample and original\*.png"
 #        )
        images_path = glob.glob(r"E:\4.FYP 3rd folder\3.Videos\1. 31 Mar 2021\1.Pics\*.png")
        # images_path = glob.glob(r"F:\FYP\10.Redmi 8 videos\3_9_2020\2\original images\1st half\original\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\2.AK_47_ 9_nov_2020\1.pic\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\1. 8_nov_2020\3\1.pics\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\3.imageProcessed\1.11_11_2020\*.png")
        # images_path = glob.glob(r"D:\1.FYP second folder\2.data set\3.imageProcessed\2\*.png")
        # images_path = glob.glob(r"F:\FYP\10.Redmi 8 videos\3_9_2020\2\original images\1st half\0.image with processed\*.png")
        # number_of_objects = 0

        # print('number_of_objects = ' + str(number_of_objects))

        # Insert here the path of your images

        # random.shuffle(images_path)

        # loop through all the images
        #  i=0
        j = 0




        for img_path in self.images_path:
            # Loading image

            # self.detect_images(self, img_path, net, classes, colors, output_layers)

            ##--------
            # Image directory
            directory = r'D:\1.FYP second folder\2.data set\1. 8_nov_2020\3\1.pics\1.detected'
            # directory = r'D:\1.FYP second folder\3.Train model and their pics\3.Check models which are correct\1.Five classes\3.Models_yolov3_custom_last (6)'

            # Change the current directory
            # to specified directory
            os.chdir(directory)

            # Filename
            filename = str(j) + '.png'
            j = j + 1
            print(j)

            # Using cv2.imwrite() method

            # Saving the image
            # cv2.imwrite(filename, img)
            ##-------------
            # key = cv2.waitKey(0)
            if (j == 600):
                # print('number_of_objects = ' + str(number_of_objects))
                break
        cv2.destroyAllWindows()

        ##------------------------------------------------


