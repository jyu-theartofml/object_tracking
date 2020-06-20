# object_tracking

The SSD object detection codes are based largely on the curriculum of Deep Learning and Computer Vision A-Z by by Hadelin de Ponteves, Kirill Eremenko. The tracking algorithm is based on Pyimagesearch's numerous blogs on multi object tracking.

The code base were written for SSD300 (input image size 300x300) and PyTorch 1.5,
 the packages <a href='https://pypi.org/project/opencv-python/'>OpenCV</a>
  and <a href='https://imageio.readthedocs.io/en/stable/installation.html'>Imageio</a> are also required. To execute `object_tracking.py`, user needs to specify input video (in mp4 format), output video name (name.mp4), and pretrained SSD300 weights. For example, `python object_tracking.py --input epic_horses.mp4 --output output.mp4 --weight  ssd300_mAP_77.43_v2.pth`. The script `posture_tracking.py` has dependencies on the directory pytorch-openpose, which was forked from <a href='https://github.com/Hzzone/pytorch-openpose'>pytorch-openpose</a>.



Please refer to blog post jyu-theartofml.github.io/posts/object_tracking.
