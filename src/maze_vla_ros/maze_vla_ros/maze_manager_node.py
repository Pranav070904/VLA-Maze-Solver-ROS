import subprocess
import sys
import os

#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VLA-Maze-Solver', 'src'))
sys.path.insert(0, '/home/pranav/maze_vla_ws/src/maze_vla_ros/VLA-Maze-Solver/src')


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray

GOAL_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

from maze_vla_ros.maze_to_sdf import wall_models, goal_model, floor_model, cell_to_world
from env_wrapper import MazeEnv

WORLD = "maze_world"
CELL_SIZE = 1.0


class MazeManager(Node):
    def __init__(self):
        super().__init__('maze_manager')
        self.env = MazeEnv(maze_size=5, render_size=64, max_steps=200)
        self.spawned = []
        self.declare_parameter('flat_materials', True)

        self.goal_pub = self.create_publisher(Float32MultiArray, '/goal_positions', GOAL_QOS)
        self.create_subscription(Bool, '/episode_done', self.on_episode_done, 10)
        self.get_logger().info("Maze manager ready, spawning first maze...")
        self.new_episode()

    def on_episode_done(self, msg: Bool):
        self.get_logger().info("Episode done, generating new maze...")
        self.new_episode()

    def new_episode(self):
        # freeze physics and the camera feed while the maze is rebuilt, so the
        # model isn't fed a half-built scene and the robot can't drift
        self.set_paused(True)
        self.clear_maze()
        self.env.reset()
        self.spawn_maze()
        self.teleport_robot_to_start()
        self.set_paused(False)
        # published last: "goals arrived" means the episode is fully set up
        self.goal_pub.publish(Float32MultiArray(data=self.goal_xy))

    def set_paused(self, paused):
        subprocess.run([
            "gz", "service", "-s", f"/world/{WORLD}/control",
            "--reqtype", "gz.msgs.WorldControl", "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000", "--req", f"pause: {str(paused).lower()}"
        ], check=False)

    def spawn(self, name, sdf, x, y, z):
        # call the gz service directly: `ros2 run ros_gz_sim create` spins up a
        # whole ROS node per model, which is slow and flaky ~20 times per reset
        sdf_escaped = sdf.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
        subprocess.run([
            "gz", "service", "-s", f"/world/{WORLD}/create",
            "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", f'sdf: "{sdf_escaped}" name: "{name}" pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}'
        ], check=False)
        self.spawned.append(name)

    def despawn(self, name):
        subprocess.run([
            "gz", "service", "-s", f"/world/{WORLD}/remove",
            "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000", "--req", f'name: "{name}" type: MODEL'
        ], check=False)

    def clear_maze(self):
        for name in self.spawned:
            self.despawn(name)
        self.spawned = []

    def spawn_maze(self):
        flat = self.get_parameter('flat_materials').value

        name, sdf, x, y, z = floor_model(self.env.grid.shape, CELL_SIZE, flat_materials=flat)
        self.spawn(name, sdf, x, y, z)

        for name, sdf, x, y, z in wall_models(self.env.grid, CELL_SIZE):
            self.spawn(name, sdf, x, y, z)

        name, sdf, red_x, red_y, z = goal_model("goal_red", *self.env.red_goal_pos, "red", CELL_SIZE, flat_materials=flat)
        self.spawn(name, sdf, red_x, red_y, z)
        name, sdf, green_x, green_y, z = goal_model("goal_green", *self.env.green_goal_pos, "green", CELL_SIZE, flat_materials=flat)
        self.spawn(name, sdf, green_x, green_y, z)
        self.goal_xy = [red_x, red_y, green_x, green_y]

    def teleport_robot_to_start(self):
        x, y = cell_to_world(*self.env.agent_pos, CELL_SIZE)
        subprocess.run([
            "gz", "service", "-s", f"/world/{WORLD}/set_pose",
            "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", f'name: "robot" position: {{x: {x}, y: {y}, z: 0.1}}'
        ], check=False)


def main():
    rclpy.init()
    node = MazeManager()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()