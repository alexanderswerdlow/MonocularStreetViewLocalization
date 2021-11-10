import cv2
import os
import imutils
import numpy as np

from config import images_dir, segmentation_model_dir

class SemanticSegmentation:
    def __init__(self):
        self.enet_neural_network = cv2.dnn.readNetFromTorch(os.path.join(segmentation_model_dir, 'enet-model.net'))
        self.class_names = (open(os.path.join(segmentation_model_dir, 'enet-classes.txt')).read().strip().split("\n"))

        self.IMG_COLOR_LIST = (
            open(os.path.join(segmentation_model_dir, 'enet-colors.txt')).read().strip().split("\n"))
        self.IMG_COLOR_LIST = [np.array(color.split(",")).astype(
            "int") for color in self.IMG_COLOR_LIST]
        self.IMG_COLOR_LIST = np.array(self.IMG_COLOR_LIST, dtype="uint8")

    def segmentImage(self, frame):
        # https://automaticaddison.com/how-to-detect-objects-using-semantic-segmentation/
        RESIZED_WIDTH = 600
        IMG_NORM_RATIO = 1 / 255.0
        ENET_DIMENSIONS = (1024, 512)
        
        input_img = imutils.resize(frame, width=RESIZED_WIDTH)
        input_img_blob = cv2.dnn.blobFromImage(frame, IMG_NORM_RATIO, ENET_DIMENSIONS, 0, swapRB=True, crop=False)
        self.enet_neural_network.setInput(input_img_blob)
        enet_neural_network_output = self.enet_neural_network.forward()

        (number_of_classes, height, width) = enet_neural_network_output.shape[1:4] 
        class_map = np.argmax(enet_neural_network_output[0], axis=0)
        
        class_map_mask = cv2.resize(class_map, (
            input_img.shape[1], input_img.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        class_legend = np.zeros(((len(self.class_names) * 25) + 25, 300, 3), dtype="uint8")
     
        # Put the class labels and colors on the legend
        for (i, (cl_name, cl_color)) in enumerate(zip(self.class_names, self.IMG_COLOR_LIST)):
            color_information = [int(color) for color in cl_color]
            cv2.putText(class_legend, cl_name, (5, (i * 25) + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(class_legend, (100, (i * 25)), (300, (i * 25) + 25),
                            tuple(color_information), -1)

        filtered_img = np.zeros_like(input_img)
        segments_filter = (class_map_mask <= 5) | (class_map_mask == 11)
        filtered_img[segments_filter] = input_img[segments_filter]

        # cv2.imshow('Segmented', filtered_img)
        # cv2.waitKey(0)
        
        return filtered_img