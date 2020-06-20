import os
os.chdir('pytorch-openpose/')

import sys
import cv2
import model
import src.util
from src.body import Body
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import torch
import imageio
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input video")
parser.add_argument("-o", "--output", required=True, help="path to output video")
parser.add_argument(
    "-w", "--weight", required=True, help="pretrained model weight in .pth file"
)

reader = imageio.get_reader(args.input)
# We get the fps frequence (frames per second).
fps = reader.get_meta_data()["fps"]
writer = imageio.get_writer(args.output, fps=fps)

body_estimation = Body(args.weight)

limbSeq = [
    [2, 3],
    [2, 6],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 9],
    [9, 10],
    [10, 11],
    [2, 12],
    [12, 13],
    [13, 14],
    [2, 1],
    [1, 15],
    [15, 17],
    [1, 16],
    [16, 18],
    [3, 17],
    [6, 18],
]
colors = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]
stickwidth = 20


def angle_cal(arm_coord, elbow_coord, hand_coord):
    a = arm_coord
    e = elbow_coord
    h = hand_coord
    ae = (e[0] - a[0], e[1] - a[1])
    he = (e[0] - h[0], e[1] - h[1])

    dot = ae[0] * he[0] + ae[1] * he[1]
    # rotate axis to align with positive direction
    cross = ae[0] * he[1] - ae[1] * he[0]

    alpha = math.atan2(cross, dot)
    degree = math.degrees(alpha)
    return degree


def detect_posture(canvas, net):
    """
    canvas is the individual frame/picture
    net is the Body model object loaded with pretrained weight
    """

    height, width = canvas.shape[:2]
    # 480 #width, height, keep the same aspect ratio
    resized_img = cv2.resize(canvas, (320, 180))
    xScale = width / 320
    yScale = height / 180
    candidate, subset = net(resized_img)
    for n in range(len(subset)):
        # find the points of interest
        armL_idx = int(subset[n][5])
        armR_idx = int(subset[n][2])

        elbowL_idx = int(subset[n][6])
        elbowR_idx = int(subset[n][3])

        handL_idx = int(subset[n][7])
        handR_idx = int(subset[n][4])

        hipL_idx = int(subset[n][11])
        hipR_idx = int(subset[n][8])
        kneeL_idx = int(subset[n][12])
        kneeR_idx = int(subset[n][9])
        footL_idx = int(subset[n][13])
        footR_idx = int(subset[n][10])
        # estimate the angles for the arms and legs
        left_arm_deg = angle_cal(
            candidate[armL_idx][0:2],
            candidate[elbowL_idx][0:2],
            candidate[handL_idx][0:2],
        )
        right_arm_deg = angle_cal(
            candidate[armR_idx][0:2],
            candidate[elbowR_idx][0:2],
            candidate[handR_idx][0:2],
        )
        left_leg_deg = angle_cal(
            candidate[hipL_idx][0:2],
            candidate[kneeL_idx][0:2],
            candidate[footL_idx][0:2],
        )
        right_leg_deg = angle_cal(
            candidate[hipR_idx][0:2],
            candidate[kneeR_idx][0:2],
            candidate[footR_idx][0:2],
        )

        head_idx = int(subset[n][1])
        headx, heady = candidate[head_idx][0:2]

        for i in range(0, 12):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0] * xScale
            X = candidate[index.astype(int), 1] * yScale
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)),
                (int(length / 2), int(stickwidth)),
                int(angle),
                0,
                360,
                1,
            )
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        # put this after the blending of canvas and cur_canvas so runner labels show up clearly
        if (0 < abs(left_arm_deg) < 100 or 0 < abs(right_arm_deg) < 100) & (
            0 < abs(left_leg_deg) < 140 or 0 < abs(right_leg_deg) < 140
        ):
            cv2.putText(
                canvas,
                "Runner",
                (int(headx * xScale) - 10, int(heady * yScale) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 105, 180),
                10,
                cv2.LINE_AA,
            )

    return canvas


for i, frame in enumerate(reader):
    posture_img = detect_posture(frame, body_estimation)
    writer.append_data(posture_img)

writer.close()
