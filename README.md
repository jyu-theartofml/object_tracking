# object_tracking

The SSD object detection codes are based largely on the curriculum of Deep Learning and Computer Vision A-Z by Hadelin de Ponteves, Kirill Eremenko. The tracking algorithm is based on ideas from Pyimagesearch's numerous blogs on object tracking.

The codes were written for SSD300 (input image size 300x300) and PyTorch 1.5,
 the packages <a href='https://pypi.org/project/opencv-python/'>OpenCV</a>
  and <a href='https://imageio.readthedocs.io/en/stable/installation.html'>Imageio</a> are also required.

* The script `object_detect.py` displays the bounding box(bbox) and the centroids in the output mp4.
* `object_tracking_line.py` displays the bbox, the centroids, and the movement of the tracked centroids in the output.
* `object_tracking_anonymous.py` is similar to `object_tracking_line.py`, except it only displays the bbox,
and the centroid tracking without showing the subject/objects to preserve privacy.
* The script `posture_tracking.py` has dependencies on the directory pytorch-openpose, which was forked from <a href='https://github.com/Hzzone/pytorch-openpose'>pytorch-openpose</a>. It overlays the skeleton of the body
key points on the frame, and detects the presence of runners. It's highly recommended to run this script on a GPU.

To execute the script, user needs to specify input video (in mp4 format), output video name (name.mp4), and pretrained SSD300 weights. For example, `python object_detect.py --input epic_horses.mp4 --output output.mp4 --weight  ssd300_mAP_77.43_v2.pth`. For `posture_tracking.py`, the pre-trained model weights can be found at this <a href='https://github.com/Hzzone/pytorch-openpose#download-the-models'>link</a>.

Please refer to this blog <a href='https://jyu-theartofml.github.io/posts/object_tracking'>post</a> for more details.
