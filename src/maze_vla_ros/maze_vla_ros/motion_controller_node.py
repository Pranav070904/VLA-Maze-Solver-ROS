import math
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy
from collections import deque
from std_msgs.msg import Int8, Bool, Float32MultiArray, String
from geometry_msgs.msg import Twist, Pose

CELL_SIZE = 1.0
# 0=up 1=down 2=left 3=right in image space. With cell_to_world (x=col, y=-row)
# and the overhead camera oriented so image-up = +y, image-right = +x:
ACTION_YAW = {0: math.pi / 2, 1: -math.pi / 2, 2: math.pi, 3: 0.0}
GOAL_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
GOAL_RADIUS = 0.5       # within half a cell counts as "reached"
# the policy can livelock, flip-flopping between two cells forever. At ~8s per
# step a 200-step timeout is ~28min, so cut an episode short once the recent
# cell history collapses to a 1- or 2-cell cycle.
LIVELOCK_WINDOW = 8
LIVELOCK_CELLS = 2
ARRIVE_TOL = 0.06       # how close to a cell centre counts as arrived
LIN_SPEED = 0.25
ANG_CLAMP = 1.0
ANG_MIN = 0.3
STALL_TIMEOUT = 1.5    # sim seconds without closing distance -> treat as blocked
WALL_BACKSTOP = 120.0  # real seconds; only trips if /clock stops, to avoid hanging


class MotionController(Node):
    def __init__(self):
        super().__init__('motion_controller')
        self.declare_parameter('max_steps', 200)  # matches MazeEnv(max_steps=200)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.done_pub = self.create_publisher(Bool, '/episode_done', 10)
        # 'red' | 'green' | 'timeout' | 'livelock', for the eval harness
        self.outcome_pub = self.create_publisher(String, '/episode_outcome', 10)
        # on_action must stay mutually exclusive: reentrant would let two
        # queued actions both pass the (unlocked) busy check and run at once.
        reentrant = ReentrantCallbackGroup()
        action_group = MutuallyExclusiveCallbackGroup()
        # depth=1 so a freed-up controller acts on the newest action, not a stale queued one
        self.create_subscription(Int8, '/agent_action', self.on_action, 1, callback_group=action_group)
        self.create_subscription(Pose, '/model/robot/pose', self.on_pose, 10, callback_group=reentrant)
        self.create_subscription(Float32MultiArray, '/goal_positions', self.on_goals, GOAL_QOS, callback_group=reentrant)
        self.pose = None
        self.busy = False
        self.goals = None   # None also means "between episodes": ignore actions
        self.steps = 0
        self.recent_cells = deque(maxlen=LIVELOCK_WINDOW)

    def on_pose(self, msg: Pose):
        self.pose = msg

    def on_goals(self, msg: Float32MultiArray):
        red_x, red_y, green_x, green_y = msg.data
        self.goals = [(red_x, red_y), (green_x, green_y)]
        self.steps = 0  # new goal positions == new episode fully set up
        self.recent_cells.clear()

    def on_action(self, msg: Int8):
        if self.busy or self.pose is None or self.goals is None:
            return
        self.busy = True
        try:
            self.execute_move(msg.data)
        finally:
            self.busy = False

    def execute_move(self, action):
        """One discrete step, mirroring MazeEnv.step()."""
        yaw = ACTION_YAW.get(action)
        if yaw is None:
            return
        cx, cy = round(self.pose.position.x), round(self.pose.position.y)
        tx, ty = cx + round(math.cos(yaw)), cy + round(math.sin(yaw))
        self.get_logger().info(f"step {self.steps} action={action} cell=({cx},{cy}) -> ({tx},{ty})")

        self.rotate_to(yaw)
        arrived = self.drive_to(tx, ty)
        if not arrived:
            self.get_logger().info("blocked, returning to cell centre")
            self.rotate_to(math.atan2(cy - self.pose.position.y, cx - self.pose.position.x))
            self.drive_to(cx, cy)
        self.steps += 1
        self.recent_cells.append((round(self.pose.position.x), round(self.pose.position.y)))

        reached = self._goal_reached()
        max_steps = self.get_parameter('max_steps').value
        if reached:
            self.get_logger().info(f"{reached} goal reached in {self.steps} steps -- resetting")
            self._end_episode(reached)
        elif self.steps >= max_steps:
            self.get_logger().info(f"timeout after {self.steps} steps -- resetting")
            self._end_episode('timeout')
        elif self._livelocked():
            self.get_logger().warn(
                f"livelocked: last {LIVELOCK_WINDOW} steps covered only "
                f"{len(set(self.recent_cells))} cell(s) -- resetting")
            self._end_episode('livelock')

    def _livelocked(self):
        return (len(self.recent_cells) == LIVELOCK_WINDOW
                and len(set(self.recent_cells)) <= LIVELOCK_CELLS)

    def _end_episode(self, outcome):
        self.goals = None   # ignore actions until maze_manager publishes new goals
        self.outcome_pub.publish(String(data=outcome))
        self.done_pub.publish(Bool(data=True))

    def _now(self):
        """Sim seconds (use_sim_time), so timeouts don't fire early when the
        real-time factor drops below 1."""
        return self.get_clock().now().nanoseconds * 1e-9

    def rotate_to(self, target_yaw, tolerance=0.05, timeout=6.0):
        twist = Twist()
        deadline = self._now() + timeout
        wall_stop = time.monotonic() + WALL_BACKSTOP
        while True:
            error = self._angle_diff(target_yaw, self._yaw())
            if abs(error) < tolerance:
                break
            if self._now() > deadline or time.monotonic() > wall_stop:
                self.get_logger().warn(f"rotate_to timed out, error={error:.2f}")
                break
            # ANG_MIN floors the speed so it can overcome wheel stiction
            w = max(ANG_MIN, min(ANG_CLAMP, 2.5 * abs(error)))
            twist.angular.z = math.copysign(w, error)
            self.cmd_pub.publish(twist)
            time.sleep(0.02)
        self.cmd_pub.publish(Twist())

    def drive_to(self, tx, ty, timeout=8.0):
        """Drive to an absolute point, steering to correct lateral error on the
        way. Returns False if it stalled (blocked) before arriving."""
        twist = Twist()
        deadline = self._now() + timeout
        wall_stop = time.monotonic() + WALL_BACKSTOP
        last_progress_t = self._now()
        last_d = self._dist_to(tx, ty)
        while True:
            d = self._dist_to(tx, ty)
            if d < ARRIVE_TOL:
                self.cmd_pub.publish(Twist())
                return True
            if d < last_d - 0.01:
                last_d, last_progress_t = d, self._now()
            if (self._now() - last_progress_t > STALL_TIMEOUT or self._now() > deadline
                    or time.monotonic() > wall_stop):
                self.cmd_pub.publish(Twist())
                self.get_logger().warn(f"drive_to stalled {d:.2f}m from target")
                return False
            heading = math.atan2(ty - self.pose.position.y, tx - self.pose.position.x)
            err = self._angle_diff(heading, self._yaw())
            twist.linear.x = LIN_SPEED if abs(err) < 0.5 else 0.0
            twist.angular.z = max(-ANG_CLAMP, min(ANG_CLAMP, 2.0 * err))
            self.cmd_pub.publish(twist)
            time.sleep(0.02)

    def _goal_reached(self):
        """Which goal the robot is standing on, if any. goals is [red, green]."""
        if not self.goals:
            return None
        for name, (gx, gy) in zip(('red', 'green'), self.goals):
            if self._dist_to(gx, gy) < GOAL_RADIUS:
                return name
        return None

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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
