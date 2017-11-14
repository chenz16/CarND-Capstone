import os

from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import rospkg


CKPT = os.path.join('light_classification', 'frozen', 'frozen_inference_graph.pb')
CLASS_TO_STATE = {
    1: TrafficLight.RED,
    2: TrafficLight.YELLOW,
    3: TrafficLight.GREEN    
}
SSD_SIZE = 600


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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # implement light color prediction
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                h, w, _ = image.shape
                image = cv2.resize(image, (int(SSD_SIZE * h/w), SSD_SIZE))
                pad = SSD_SIZE - image.shape[0]
                image = np.pad(image, ((pad, 0), (0, 0), (0, 0)), 'constant')
                image_expanded = np.expand_dims(image, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

        total_scores = np.zeros(3, dtype=np.float32)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        if num > 0:
            for cls, score in zip(classes, scores):
                if score > 0.5:
                    total_scores[int(cls)-1] += score
            prediction = CLASS_TO_STATE[np.argmax(total_scores) + 1]
        else:
            prediction = TrafficLight.UNKNOWN 

        return prediction
