import torch
from torch.autograd import Variable
import cv2
import numpy as np
import argparse
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
from pyimagesearch.centroidtracker import CentroidTracker
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input video")
parser.add_argument("-o", "--output", required=True, help="path to output video")
parser.add_argument(
    "-w", "--weight", required=True, help="pretrained model weight in .pth file"
)

args = parser.parse_args()

reader = imageio.get_reader(args.input)
# We get the fps frequence (frames per second).
fps = reader.get_meta_data()["fps"]
# We create an output video with this same fps frequence.
writer = imageio.get_writer(args.output, fps=fps)

# Creating the SSD neural network
net = build_ssd("test")  # We create an object that is our neural network ssd.
net.load_state_dict(torch.load(args.weight, map_location=lambda storage, loc: storage))
# We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

# Defining a function that will do the detections and object class tracking


def tracking(frame, net, transform, f_idx, track_pts, base_frame):
    """
    frame: individual frame
    net: pretrained object detector mdoel
    transform: function to transform image of the frame
    f_idx: frame count
    track_pts: deque collection of centroids
    base_frame: use first frame
    """
    height, width = frame.shape[:2]  # We get the height and the width of the frame.
    frame_t = transform(frame)[0]  # We apply the transformation to our frame.
    # We convert the frame into a torch tensor.
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = x.unsqueeze(0)  # We add a fake dimension corresponding to the batch.
    with torch.no_grad():
        # We feed the neural network ssd with the image and we get the output y.
        y = net(x)
    # the dimension is (N,i,j,b), i is the class, j is the number of detecttion, b axis is confidence and bounding box coordinates
    detections = y.data
    # We create a tensor object of dimensions [width, height, width, height].
    scale = torch.Tensor([width, height, width, height])

    # keep track of centroid points for this individual frame
    centroid_pts = {f_idx: {}}
    color_ls = [
        (79, 111, 243),
        (129, 199, 190),
        (86, 160, 82),
        (181, 190, 183),
        (49, 90, 93),
        (142, 132, 21),
        (69, 116, 8),
        (9, 68, 193),
        (126, 152, 167),
        (103, 95, 90),
        (98, 117, 184),
        (186, 112, 111),
        (234, 54, 27),
        (90, 218, 83),
        (134, 185, 119),
        (174, 1, 141),
        (2, 222, 238),
        (219, 86, 39),
        (155, 151, 186),
        (192, 221, 232),
        (174, 149, 85),
    ]

    for i in range(detections.size(1)):  # For every class:
        # We initialize the loop variable j that will correspond to the occurrences of the class.
        j = 0
        rects = []
        # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
        while detections[0, i, j, 0] >= 0.65:
            # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(
                base_frame,
                (int(pt[0]), int(pt[1])),
                (int(pt[2]), int(pt[3])),
                color_ls[i],
                3,
            )
            # save the bounding box coordinates (startX, startY, endX, endY)
            rects.append((int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3])))
            j += 1
            objects = ct.update(rects)

            # loop over the tracked centroids
            centroid_pts[f_idx][i] = {}
            for (objectID, centroid) in objects.items():
                # add centroid pt to deque list for each class
                w = {objectID: centroid}
                centroid_pts[f_idx][i].update(w)

                #  draw the centroid of the object on the output frame
                cv2.circle(base_frame, (centroid[0], centroid[1]), 15, color_ls[i], -1)
                text = labelmap[i - 1] + "_" + str(objectID)
                cv2.putText(
                    base_frame,
                    text,
                    (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    track_pts.appendleft(centroid_pts)  # update the deque list
    # loop through track_pts list and plot the "contrails" of the object_tracking
    for i in range(len(track_pts) - 1):
        frame_idx = list(track1[i].keys())[0]
        current = track1[i][frame_idx]
        for class_idx, v in current.items():
            for label_id, coord in v.items():
                try:
                    y = track1[i][frame_idx][class_idx][label_id]
                    y_t = track1[i + 1][frame_idx - 1][class_idx][label_id]
                    # draw the line
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    cv2.line(
                        base_frame, tuple(y_t), tuple(y), color_ls[class_idx], thickness
                    )
                except KeyError:
                    pass

    return base_frame, track_pts


baseframe = reader.get_data(0)
track1 = deque(maxlen=60)  # initialize list
ct = CentroidTracker()
base_img = []
for i, frame in enumerate(reader):
    background = baseframe.copy()
    base_img.append(background)
    # a trick to clear out the base frame
    g = base_img.pop()
    out_frame, track1 = tracking(frame, net, transform, i, track1, g)
    writer.append_data(out_frame)
    print(i)
writer.close()  # We close the video
