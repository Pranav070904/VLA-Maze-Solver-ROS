from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('maze_vla_ros')

    world_arg = DeclareLaunchArgument(
        'world', default_value='maze_world.sdf',
        description='World filename under worlds/ (e.g. maze_world_shadows.sdf) for perturbation sweeps'
    )
    world_path = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('world')])

    flat_materials_arg = DeclareLaunchArgument(
        'flat_materials', default_value='true',
        description='false lets floor/goal materials respond to lighting -- '
                     'needed for shadow/off-angle-light perturbation sweeps to have any visible effect'
    )

    threshold_arg = DeclareLaunchArgument(
        'threshold', default_value='false',
        description='true snaps every observed pixel to the nearest of the 5 training '
                     'colors before it reaches the model -- a mitigation to test against '
                     'shadow/noise/lighting perturbations'
    )

    floor_texture_arg = DeclareLaunchArgument(
        'floor_texture', default_value='false',
        description='true tiles a checker texture onto the floor instead of flat white -- '
                     'stresses wall/floor edge detection independent of lighting. '
                     'The floor material auto-disables its own emissive when this is set, '
                     'so it does not need flat_materials=false to be visible.'
    )

    floor_texture_path_arg = DeclareLaunchArgument(
        'floor_texture_path',
        default_value='/home/pranav/maze_vla_ws/src/maze_vla_ros/worlds/textures/floor_checker.png',
        description='absolute path to the image used for floor_texture:=true -- swap in any '
                     'image without touching code or rebuilding'
    )

    # start paused so the robot doesn't fall before maze_manager spawns the floor
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', world_path],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/overhead_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/robot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # sim clock: motion_controller's timeouts must be in sim time, or a
            # real-time factor below 1 makes them fire early and look like stalls
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # ground-truth pose: odometry is wheel dead-reckoning and does not
            # follow the set_pose teleport that maze_manager does on every reset
            '/model/robot/pose@geometry_msgs/msg/Pose[gz.msgs.Pose',
        ],
        output='screen'
    )

    # give gz_sim a moment to come up before starting the ROS-side nodes
    image_to_maze = TimerAction(period=3.0, actions=[
        Node(
            package='maze_vla_ros', executable='image_to_maze', output='screen',
            parameters=[{'threshold': ParameterValue(LaunchConfiguration('threshold'), value_type=bool)}]
        )
    ])
    vla_inference = TimerAction(period=3.0, actions=[
        Node(package='maze_vla_ros', executable='vla_inference', output='screen')
    ])
    motion_controller = TimerAction(period=3.0, actions=[
        Node(
            package='maze_vla_ros', executable='motion_controller', output='screen',
            parameters=[{'use_sim_time': True}]
        )
    ])
    maze_manager = TimerAction(period=5.0, actions=[
        Node(
            package='maze_vla_ros', executable='maze_manager', output='screen',
            parameters=[{
                'flat_materials': ParameterValue(LaunchConfiguration('flat_materials'), value_type=bool),
                'floor_texture': ParameterValue(LaunchConfiguration('floor_texture'), value_type=bool),
                'floor_texture_path': LaunchConfiguration('floor_texture_path'),
            }]
        )
    ])

    return LaunchDescription([
        world_arg,
        flat_materials_arg,
        threshold_arg,
        floor_texture_arg,
        floor_texture_path_arg,
        gz_sim,
        bridge,
        image_to_maze,
        vla_inference,
        motion_controller,
        maze_manager,
    ])