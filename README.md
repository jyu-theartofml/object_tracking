# object_tracking

The SSD object detection codes are based largely on the curriculum of Deep Learning and Computer Vision A-Z by by Hadelin de Ponteves, Kirill Eremenko. The tracking algorithm is based on Pyimagesearch's numerous blogs on multi object tracking.

The SSD model code base runs for PyTorch 1.5 or above. To execute `object_tracking.py`, user needs to specify input video (in mp4 format), output video name (name.mp4), and pretrained SSD weights. For example, `python object_tracking.py --input epic_horses.mp4 --output output.mp4 --weight  ssd300_mAP_77.43_v2.pth`.
