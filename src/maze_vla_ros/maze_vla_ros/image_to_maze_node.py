import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

BLACK_THRESH = 40
CHROMA_THRESH = 60
HUE_TOL = 8
MAX_ENTITY_AREA = 150
WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
ENTITY_COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
TARGET_HUES = np.array([0, 60, 120])


def snap_to_palette(frame):
    px = frame.reshape(-1, 3).astype(np.int32)
    is_black = px.max(axis=1) < BLACK_THRESH
    chroma = px.max(axis=1) - px.min(axis=1)
    is_grayish = chroma < CHROMA_THRESH

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0].astype(np.int32)
    hue_dist = np.abs(hue[:, None] - TARGET_HUES[None, :])
    hue_dist = np.minimum(hue_dist, 180 - hue_dist)
    nearest_hue = hue_dist.argmin(axis=1)
    is_colored = ~is_grayish & (hue_dist.min(axis=1) <= HUE_TOL)

    out = np.tile(WHITE, (px.shape[0], 1))
    out[is_colored] = ENTITY_COLORS[nearest_hue[is_colored]]
    out[is_black] = BLACK
    out = out.reshape(frame.shape).astype(np.uint8)
    return _drop_oversized_entities(out)


def _drop_oversized_entities(out):
    """Revert entity-coloured regions too big to be a goal/agent back to floor."""
    for colour in ENTITY_COLORS:
        mask = np.all(out == colour, axis=2).astype(np.uint8)
        if not mask.any():
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] > MAX_ENTITY_AREA:
                out[labels == i] = WHITE
    return out


class ImageToMaze(Node):
    def __init__(self):
        super().__init__('image_to_maze')
        self.declare_parameter('threshold', True)
        self.bridge = CvBridge()
        self.create_subscription(Image, '/overhead_camera/image_raw', self.cb, 10)
        self.pub = self.create_publisher(Image, '/maze_obs', 10)
        self.debug_pub = self.create_publisher(Image, '/maze_obs_thresholded', 10)

    def _to_msg(self, frame, header):
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
