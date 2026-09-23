import sys
sys.path.insert(0, '/home/pranav/maze_vla_ws/src/maze_vla_ros/VLA-Maze-Solver/src')
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8, String
from cv_bridge import CvBridge
import torch, clip

from transformer import VLA


class VLAInference(Node):
    def __init__(self):
        super().__init__('vla_inference')
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = VLA().to(self.device)
        ckpt_path = '/home/pranav/maze_vla_ws/src/maze_vla_ros/VLA-Maze-Solver/checkpoints/bc_init_v3_epoch20.pt'
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.agent.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.agent.eval()

        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        # no actions are published until an instruction arrives on /instruction
        self.text_emb = None

        self.create_subscription(Image, '/maze_obs', self.on_obs, 10)
        self.create_subscription(String, '/instruction', self.on_instruction, 10)
        self.pub = self.create_publisher(Int8, '/agent_action', 10)

        self.get_logger().info("VLA inference node ready -- waiting for /instruction")

    def on_instruction(self, msg: String):
        with torch.no_grad():
            self.text_emb = self.clip_model.encode_text(
                clip.tokenize([msg.data]).to(self.device)
            ).float()
        self.get_logger().info(f"Instruction set: \"{msg.data}\"")

    def on_obs(self, msg: Image):
        if self.text_emb is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        s = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            logits, _ = self.agent(s, self.text_emb)
            action = int(torch.argmax(logits, dim=-1).item())

        self.pub.publish(Int8(data=action))


def main():
    rclpy.init()
    rclpy.spin(VLAInference())
    rclpy.shutdown()


if __name__ == '__main__':
    main()