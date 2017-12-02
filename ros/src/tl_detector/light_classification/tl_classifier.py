import os

from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import rospkg
from PIL import Image

import visualization_utils

# SSD Mobilenet
CKPT = os.path.join('light_classification', 'frozen', 'frozen_inference_graph.pb')
CLASS_TO_STATE = {
    1: TrafficLight.RED,
    2: TrafficLight.YELLOW,
    3: TrafficLight.GREEN
}
SSD_SIZE = 600
SCORE_THRESH = 0.3

# Debugging
LOG_DIR = '/home/btamm/Documents/udacity/term3/CarND-Capstone_old/results/sim_results'
CLASSIFICATION_LOG = '/home/btamm/Documents/udacity/term3/CarND-Capstone_old/results/classification_log.txt'
STATE_TO_COLOR = {
    TrafficLight.RED: 'Red',
    TrafficLight.YELLOW: 'Yellow',
    TrafficLight.GREEN: 'Green',
    TrafficLight.UNKNOWN: 'Unknown',
}
CLASS_DICTS = {int(1): {'id': int(1), 'name': 'RED'},
               int(2): {'id': int(2), 'name': 'YELLOW'},
               int(3): {'id': int(3), 'name': 'GREEN'}}

# # Mobilenet
# CKPT = os.path.join('light_classification', 'frozen', 'mobilenet_tl_frozen.pb')


class TLClassifier(object):
    def __init__(self):
        rp = rospkg.RosPack()
        module_dir = rp.get_path('tl_detector')
        path_to_ckpt = os.path.join(module_dir, CKPT)
        # load classifier
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.log_count = 0

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # implement light color prediction
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # resize and pad image to SSD shape
                h, w, _ = image.shape
                image = cv2.resize(image, (SSD_SIZE, int(SSD_SIZE * h/w)))
                pad = SSD_SIZE - image.shape[0]
                image = np.pad(image, ((pad, 0), (0, 0), (0, 0)), 'constant')
                image_expanded = np.expand_dims(image, axis=0)

                # run inference
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(int)

        # create overlay image for debugging
        draw_img = visualization_utils.visualize_boxes_and_labels_on_image_array(image.copy(),
                                                  boxes,
                                                  classes,
                                                  scores,
                                                  CLASS_DICTS,
                                                  use_normalized_coordinates=True,
                                                  min_score_thresh=SCORE_THRESH)
        self.log_count += 1
        fname = 'result_{0:0>5}.jpg'.format(self.log_count)
        image_file = os.path.join(LOG_DIR, fname)
        draw_img = Image.fromarray(draw_img)
        draw_img.save(image_file)

        # get overall prediction
        total_scores = np.zeros(3, dtype=np.float32)
        if num > 0:
            for cls, score in zip(classes, scores):
                if score > SCORE_THRESH:
                    total_scores[int(cls)-1] += score
            prediction = CLASS_TO_STATE[np.argmax(total_scores) + 1]
        else:
            prediction = TrafficLight.UNKNOWN

        # log overall classifications
        data_line = '{}, {}'.format(fname, STATE_TO_COLOR[prediction])
        with open(CLASSIFICATION_LOG, 'a+') as out_file:
            out_file.write(data_line + os.linesep)

        return prediction
