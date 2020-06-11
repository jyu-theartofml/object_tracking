import torch
from torch.autograd import Variable
import cv2
import numpy as np
import argparse
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio
from pyimagesearch.centroidtracker import CentroidTracker

parser= argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True,
	help="path to input video")
parser.add_argument("-o", "--output", required=True,
	help="path to output video")
parser.add_argument("-w", "--weight", required=True,
  	help="pretrained model weight in .pth file")

args = parser.parse_args()

reader = imageio.get_reader(args.input)
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer(args.output, fps = fps) # We create an output video with this same fps frequence.

# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load(args.weight, map_location = lambda storage, loc: storage))
 # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

ct = CentroidTracker()

# Defining a function that will do the detections and object class tracking
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0] # We apply the transformation to our frame.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame into a torch tensor.
    x = x.unsqueeze(0) # We add a fake dimension corresponding to the batch.
    # We feed the neural network ssd with the image and we get the output y.
    with torch.no_grad():
        y = net(x)
	# We create the detections tensor contained in the output y,
	# the dimension is (N,i,j,b), i is the class, j is the number of detecttion, b axis is confidence and bounding box coordinates
    detections = y.data
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height].

    for i in range(detections.size(1)): # For every class:
      j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
      rects=[]
      while detections[0, i, j, 0] >= 0.5: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
          pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
          cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object.

          # save the bounding box coordinates
          rects.append((int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]))) #(startX, startY, endX, endY)
          j += 1
          objects = ct.update(rects)
          # loop over the tracked objects
          for (objectID, centroid) in objects.items():
            # draw the centroid of the object on the output frame and put text on the frame
            text = labelmap[i-1]+'_'+str(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			      cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2,cv2.LINE_AA)
            cv2.circle(frame, (centroid[0], centroid[1]), 10, (0, 255, 0), -1)
    return frame

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i) 
writer.close() # We close the video
