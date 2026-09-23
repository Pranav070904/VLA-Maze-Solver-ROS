import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# must match env_wrapper.py's MazeEnv.COLOR_MAP exactly -- the model was
# trained on frames made of only these five flat colors.
BLACK_THRESH = 40   # every channel below this -> wall
CHROMA_THRESH = 60  # max-min channel spread below this -> gray/white, not a colored entity
WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
ENTITY_COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)  # indexed by argmax channel


def snap_to_palette(frame):
    px = frame.reshape(-1, 3).astype(np.int32)
    is_black = px.max(axis=1) < BLACK_THRESH
    # nearest-RGB-distance is wrong here: a dark shadow gray like (40,40,40)
    # is numerically CLOSER to red (255,0,0) than to white (255,255,255) --
    # only 2 channels differ vs 3. Classify by chroma instead: a grayish
    # pixel (low max-min spread) is floor/white regardless of how dark the
    # shadow made it; only a clearly saturated pixel is a colored entity.
    chroma = px.max(axis=1) - px.min(axis=1)
    is_colored = chroma >= CHROMA_THRESH
    dominant_channel = px.argmax(axis=1)

    out = np.tile(WHITE, (px.shape[0], 1))
    out[is_colored] = ENTITY_COLORS[dominant_channel[is_colored]]
    out[is_black] = BLACK
    return out.reshape(frame.shape).astype(np.uint8)


class ImageToMaze(Node):
    def __init__(self):
        super().__init__('image_to_maze')
        self.declare_parameter('threshold', True)
        self.bridge = CvBridge()
        self.create_subscription(Image, '/overhead_camera/image_raw', self.cb, 10)
        self.pub = self.create_publisher(Image, '/maze_obs', 10)
        # always published, regardless of the threshold param, so the
        # thresholded view can be inspected (rqt_image_view) independent of
        # whether it's actually the thing being fed to the model
        self.debug_pub = self.create_publisher(Image, '/maze_obs_thresholded', 10)

    def _to_msg(self, frame, header):
        # cv_bridge 4.1.0's cv2_to_imgmsg has a broken encoding_to_cvtype2/cvtype_to_name
        # mapping for multi-channel encodings (raises KeyError for 'rgb8'), so build the
        # message via passthrough and set the encoding directly instead.
        out = self.bridge.cv2_to_imgmsg(frame, encoding='passthrough')
        out.encoding = 'rgb8'
        out.header = header
        return out

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        thresholded = snap_to_palette(resized)
        self.debug_pub.publish(self._to_msg(thresholded, msg.header))
        out_frame = thresholded if self.get_parameter('threshold').value else resized
        self.pub.publish(self._to_msg(out_frame, msg.header))


def main():
    rclpy.init()
    rclpy.spin(ImageToMaze())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
