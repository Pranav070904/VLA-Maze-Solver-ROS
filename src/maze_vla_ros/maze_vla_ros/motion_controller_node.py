import math
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Int8, Bool, Float32MultiArray
from geometry_msgs.msg import Twist, Pose

CELL_SIZE = 1.0
# 0=up 1=down 2=left 3=right in image space. With cell_to_world (x=col, y=-row)
# and the overhead camera oriented so image-up = +y, image-right = +x:
ACTION_YAW = {0: math.pi / 2, 1: -math.pi / 2, 2: math.pi, 3: 0.0}
GOAL_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
GOAL_RADIUS = 0.5       # within half a cell counts as "reached"
MAX_STEPS = 200         # matches MazeEnv(max_steps=200)
ARRIVE_TOL = 0.06       # how close to a cell centre counts as arrived
LIN_SPEED = 0.25
ANG_CLAMP = 1.0
ANG_MIN = 0.3


class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.done_pub = self.create_publisher(Bool, '/episode_done', 10)
        # on_pose/on_goals are reentrant so they keep getting serviced on the
        # other executor thread while on_action is blocked inside execute_move.
        # on_action itself must NOT be reentrant: two queued /agent_action
        # messages could otherwise both invoke on_action concurrently, both
        # read self.busy==False before either sets it (no lock), and both
        # call execute_move at once -- two threads fighting over /cmd_vel
        # simultaneously. MutuallyExclusive guarantees at most one at a time.
        reentrant = ReentrantCallbackGroup()
        action_group = MutuallyExclusiveCallbackGroup()
        # depth=1: don't let a backlog of stale actions build up while busy --
        # when we free up, act on the newest available action, not an ~1s-old one
        self.create_subscription(Int8, '/agent_action', self.on_action, 1, callback_group=action_group)
        self.create_subscription(Pose, '/model/robot/pose', self.on_pose, 10, callback_group=reentrant)
        self.create_subscription(Float32MultiArray, '/goal_positions', self.on_goals, GOAL_QOS, callback_group=reentrant)
        self.pose = None
        self.busy = False
        self.goals = None   # None also means "between episodes": ignore actions
        self.steps = 0

    def on_pose(self, msg: Pose):
        self.pose = msg

    def on_goals(self, msg: Float32MultiArray):
        red_x, red_y, green_x, green_y = msg.data
        self.goals = [(red_x, red_y), (green_x, green_y)]
        self.steps = 0  # new goal positions == new episode fully set up

    def on_action(self, msg: Int8):
        if self.busy or self.pose is None or self.goals is None:
            return
        self.busy = True
        try:
            self.execute_move(msg.data)
        finally:
            self.busy = False

    # ------------------------------------------------------------------
    # one discrete step, mirroring MazeEnv.step()
    # ------------------------------------------------------------------

    def execute_move(self, action):
        yaw = ACTION_YAW.get(action)
        if yaw is None:
            return
        # snap to the grid: cell centres are at integer world coords
        cx, cy = round(self.pose.position.x), round(self.pose.position.y)
        tx, ty = cx + round(math.cos(yaw)), cy + round(math.sin(yaw))
        self.get_logger().info(f"step {self.steps} action={action} cell=({cx},{cy}) -> ({tx},{ty})")

        self.rotate_to(yaw)
        arrived = self.drive_to(tx, ty)
        if not arrived:
            # wall bump: env leaves the agent where it was, so return to the
            # cell centre we started from and let the model try again
            self.get_logger().info("blocked, returning to cell centre")
            self.rotate_to(math.atan2(cy - self.pose.position.y, cx - self.pose.position.x))
            self.drive_to(cx, cy)
        self.steps += 1

        if self._at_a_goal():
            self.get_logger().info(f"goal reached in {self.steps} steps -- resetting")
            self._end_episode()
        elif self.steps >= MAX_STEPS:
            self.get_logger().info(f"timeout after {self.steps} steps -- resetting")
            self._end_episode()

    def _end_episode(self):
        self.goals = None   # ignore actions until maze_manager publishes new goals
        self.done_pub.publish(Bool(data=True))

    # ------------------------------------------------------------------
    # low-level motion
    # ------------------------------------------------------------------

    def rotate_to(self, target_yaw, tolerance=0.05, timeout=6.0):
        twist = Twist()
        deadline = time.monotonic() + timeout
        while True:
            error = self._angle_diff(target_yaw, self._yaw())
            if abs(error) < tolerance:
                break
            if time.monotonic() > deadline:
                self.get_logger().warn(f"rotate_to timed out, error={error:.2f}")
                break
            # proportional, but never below a floor that can overcome wheel stiction
            w = max(ANG_MIN, min(ANG_CLAMP, 2.5 * abs(error)))
            twist.angular.z = math.copysign(w, error)
            self.cmd_pub.publish(twist)
            time.sleep(0.02)
        self.cmd_pub.publish(Twist())

    def drive_to(self, tx, ty, timeout=8.0):
        """Drive to an absolute point, steering to correct lateral error on the
        way. Returns False if it stalled (blocked) before arriving."""
        twist = Twist()
        deadline = time.monotonic() + timeout
        last_progress_t = time.monotonic()
        last_d = self._dist_to(tx, ty)
        while True:
            d = self._dist_to(tx, ty)
            if d < ARRIVE_TOL:
                self.cmd_pub.publish(Twist())
                return True
            if d < last_d - 0.01:
                last_d, last_progress_t = d, time.monotonic()
            if time.monotonic() - last_progress_t > 1.5 or time.monotonic() > deadline:
                self.cmd_pub.publish(Twist())
                self.get_logger().warn(f"drive_to stalled {d:.2f}m from target")
                return False
            heading = math.atan2(ty - self.pose.position.y, tx - self.pose.position.x)
            err = self._angle_diff(heading, self._yaw())
            twist.linear.x = LIN_SPEED if abs(err) < 0.5 else 0.0
            twist.angular.z = max(-ANG_CLAMP, min(ANG_CLAMP, 2.0 * err))
            self.cmd_pub.publish(twist)
            time.sleep(0.02)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _at_a_goal(self):
        if not self.goals:
            return False
        return any(self._dist_to(gx, gy) < GOAL_RADIUS for gx, gy in self.goals)

    def _dist_to(self, x, y):
        return math.hypot(self.pose.position.x - x, self.pose.position.y - y)

    def _yaw(self):
        q = self.pose.orientation
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _angle_diff(target, current):
        return (target - current + math.pi) % (2 * math.pi) - math.pi


def main():
    rclpy.init()
    node = MotionController()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
