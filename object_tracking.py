import torch
from torch.autograd import Variable
import cv2
import numpy as np
import argparse
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
from pyimagesearch.centroidtracker import CentroidTracker

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input video")
parser.add_argument("-o", "--output", required=True, help="path to output video")
parser.add_argument(
    "-w", "--weight", required=True, help="pretrained model weight in .pth file"
)

args = parser.parse_args()

reader = imageio.get_reader(args.input)
# get the fps frequence (frames per second).
fps = reader.get_meta_data()["fps"]
# create an output video with this same fps frequence.
writer = imageio.get_writer(args.output, fps=fps)

# Creating the SSD neural network
net = build_ssd("test")
net.load_state_dict(torch.load(args.weight, map_location=lambda storage, loc: storage))
# We create an object of the BaseTransform class,to transform/normalize image arrays for the neural network.
transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

# Defining a function that will do the detections and object class tracking


def detect(frame, net, transform):
    """
    frame: individual frame
    net: pretrained object detector model
    transform: function to transform image of the frame
    """
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = x.unsqueeze(0)
    # We feed the neural network ssd with the image to get prediction output.
    with torch.no_grad():
        y = net(x)
    # the dimension is (N,i,j,b), i is the class, j is the number of detection (200), b axis is confidence and bounding box coordinates
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    for i in range(detections.size(1)):
        j = 0
        rects = []
        # for each class, filter for bbox with confidence score >=0.6.
        while detections[0, i, j, 0] >= 0.6:
            # SSD returns the points at the upper left and the lower right of the bbox
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(
                frame,
                (int(pt[0]), int(pt[1])),
                (int(pt[2]), int(pt[3])),
                (255, 0, 0),
                2,
            )

            # save the bounding box coordinates (startX, startY, endX, endY)
            rects.append((int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])))
            j += 1
            objects = ct.update(rects)
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # draw the centroid of the object on the output frame and put text on the frame
                text = labelmap[i - 1] + "_" + str(objectID)
                cv2.putText(
                    frame,
                    text,
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(frame, (centroid[0], centroid[1]), 10, (0, 255, 0), -1)
    return frame

ct = CentroidTracker()

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()
