"""Launch the hierarchical BPX VLA + Epoch-6 SAC navigation stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression


def enabled_and_mode(enabled, mode, expected):
    return IfCondition(
        PythonExpression(
            [
                "'",
                enabled,
                "'.lower() in ('true', '1', 'yes') and '",
                mode,
                f"' == '{expected}'",
            ]
        )
    )


def generate_launch_description():
    default_drl_workspace = os.environ.get(
        "DRLNAV_BASE_PATH", os.path.expanduser("~/DRL-Robot-Navigation-ROS2")
    )
    default_vla_workspace = default_drl_workspace

    drl_workspace = LaunchConfiguration("drl_workspace")
    vla_workspace = LaunchConfiguration("vla_workspace")
    target_mode = LaunchConfiguration("target_mode")
    start_gazebo = LaunchConfiguration("start_gazebo")
    start_random_target = LaunchConfiguration("start_random_target")
    start_random_obstacles = LaunchConfiguration("start_random_obstacles")
    start_demo4 = LaunchConfiguration("start_demo4")
    start_demo5 = LaunchConfiguration("start_demo5")
    start_demo6 = LaunchConfiguration("start_demo6")
    start_task_adapter = LaunchConfiguration("start_task_adapter")
    vlm_model = LaunchConfiguration("vlm_model")
    enable_color_fallback = LaunchConfiguration("enable_color_fallback")
    policy_speed_scale = LaunchConfiguration("policy_speed_scale")
    enable_safety_shield = LaunchConfiguration("enable_safety_shield")
    internal_safety = LaunchConfiguration("internal_safety")
    search_angular_speed = LaunchConfiguration("search_angular_speed")
    search_rotate_duration = LaunchConfiguration("search_rotate_duration")
    search_observe_duration = LaunchConfiguration("search_observe_duration")
    search_duration = LaunchConfiguration("search_duration")
    episode_timeout = LaunchConfiguration("episode_timeout")
    demo5_detect_rate = LaunchConfiguration("demo5_detect_rate")

    source_directory = PathJoinSubstitution(
        [drl_workspace, "src", "drl_navigation_ros2"]
    )
    python_path = [
        drl_workspace,
        "/.python_deps:",
        drl_workspace,
        "/src:",
        drl_workspace,
        "/src/drl_navigation_ros2:",
        os.environ.get("PYTHONPATH", ""),
    ]
    vla_python_path = [
        vla_workspace,
        "/src/vla_bridge:",
        os.environ.get("PYTHONPATH", ""),
    ]

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("turtlebot3_gazebo"),
                "launch",
                "ros2_drl_vla.launch.py",
            )
        ),
        condition=IfCondition(start_gazebo),
    )

    inference = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution([source_directory, "inference.py"]),
            "--policy-speed-scale",
            policy_speed_scale,
            "--cmd-vel-topic",
            PythonExpression(
                [
                    "'/policy_cmd_vel' if '",
                    enable_safety_shield,
                    "'.lower() in ('true', '1', 'yes') else '/cmd_vel'",
                ]
            ),
            "--internal-safety",
            internal_safety,
            "--search-angular-speed",
            search_angular_speed,
            "--search-rotate-duration",
            search_rotate_duration,
            "--search-observe-duration",
            search_observe_duration,
        ],
        additional_env={"PYTHONPATH": python_path},
        output="screen",
    )
    safety_shield = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution([source_directory, "safety_shield.py"]),
        ],
        additional_env={"PYTHONPATH": python_path},
        condition=IfCondition(enable_safety_shield),
        output="screen",
    )
    supervisor = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution([source_directory, "vla_agent_supervisor.py"]),
            "--ros-args",
            "-p",
            ["target_mode:=", target_mode],
            "-p",
            ["wait_for_environment_ready:=", start_random_obstacles],
            "-p",
            ["search_duration:=", search_duration],
        ],
        additional_env={"PYTHONPATH": python_path},
        output="screen",
    )
    random_target = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [source_directory, "random_red_box_manager.py"]
            ),
            "--ros-args",
            "-p",
            ["wait_for_environment_ready:=", start_random_obstacles],
            "-p",
            ["episode_timeout:=", episode_timeout],
        ],
        additional_env={"PYTHONPATH": python_path},
        condition=IfCondition(start_random_target),
        output="screen",
    )
    random_obstacles = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [source_directory, "random_obstacle_manager.py"]
            ),
        ],
        additional_env={"PYTHONPATH": python_path},
        condition=IfCondition(start_random_obstacles),
        output="screen",
    )
    vision_bridge = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [source_directory, "vla_detection_goal_bridge.py"]
            ),
        ],
        additional_env={"PYTHONPATH": python_path},
        condition=IfCondition(
            PythonExpression(["'", target_mode, "' == 'vision'"])
        ),
        output="screen",
    )
    demo6_bridge = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution([source_directory, "vla_target_bridge.py"]),
        ],
        additional_env={"PYTHONPATH": python_path},
        condition=IfCondition(
            PythonExpression(["'", target_mode, "' == 'demo6'"])
        ),
        output="screen",
    )
    demo4 = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [
                    vla_workspace,
                    "src",
                    "vla_bridge",
                    "vla_bridge",
                    "demo4_test_batch.py",
                ]
            ),
            "--ros-args",
            "-p",
            "rgb_topic:=/camera/image_raw",
            "-p",
            ["model:=", vlm_model],
        ],
        additional_env={"PYTHONPATH": vla_python_path},
        condition=IfCondition(start_demo4),
        output="screen",
    )
    demo5 = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [
                    vla_workspace,
                    "src",
                    "vla_bridge",
                    "vla_bridge",
                    "demo5_ros_detector.py",
                ]
            ),
            "--ros-args",
            "-p",
            "rgb_topic:=/camera/image_raw",
            "-p",
            ["enable_color_fallback:=", enable_color_fallback],
            "-p",
            ["vlm_model:=", vlm_model],
            "-p",
            ["detect_rate:=", demo5_detect_rate],
        ],
        additional_env={"PYTHONPATH": vla_python_path},
        # Both the lidar-based vision path and Demo6 consume Demo5 bboxes.
        condition=IfCondition(start_demo5),
        output="screen",
    )
    demo6 = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [
                    vla_workspace,
                    "src",
                    "vla_bridge",
                    "vla_bridge",
                    "demo6_vla_coord_node.py",
                ]
            ),
            "--ros-args",
            "-p",
            "target_frame:=odom",
        ],
        additional_env={"PYTHONPATH": vla_python_path},
        condition=enabled_and_mode(start_demo6, target_mode, "demo6"),
        output="screen",
    )
    task_adapter = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [
                    vla_workspace,
                    "src",
                    "vla_bridge",
                    "vla_bridge",
                    "vla_agent_node.py",
                ]
            ),
        ],
        additional_env={"PYTHONPATH": vla_python_path},
        condition=IfCondition(start_task_adapter),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "drl_workspace", default_value=default_drl_workspace
            ),
            DeclareLaunchArgument(
                "vla_workspace", default_value=default_vla_workspace
            ),
            DeclareLaunchArgument(
                "target_mode",
                default_value="vision",
                description="Exactly one target path: vision or demo6",
                choices=["vision", "demo6"],
            ),
            DeclareLaunchArgument("start_gazebo", default_value="true"),
            DeclareLaunchArgument("start_random_target", default_value="true"),
            DeclareLaunchArgument("start_random_obstacles", default_value="true"),
            DeclareLaunchArgument("start_demo4", default_value="true"),
            DeclareLaunchArgument("start_demo5", default_value="true"),
            DeclareLaunchArgument("start_demo6", default_value="false"),
            DeclareLaunchArgument("start_task_adapter", default_value="true"),
            DeclareLaunchArgument(
                "policy_speed_scale",
                default_value="0.60",
                description=(
                    "Scale deterministic SAC linear/angular commands; "
                    "0.60 makes navigation about 1.7 times slower"
                ),
            ),
            DeclareLaunchArgument(
                "enable_safety_shield",
                default_value="false",
                description=(
                    "Route SAC actions through the independent lidar safety shield"
                ),
            ),
            DeclareLaunchArgument(
                "internal_safety",
                default_value="true",
                description=(
                    "Enable legacy safety inside SAC; set false for shield ablations"
                ),
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "search_angular_speed",
                default_value="0.35",
                description="Angular speed during each bounded search turn",
            ),
            DeclareLaunchArgument(
                "search_rotate_duration",
                default_value="60.0",
                description="Seconds rotating in each bounded search cycle",
            ),
            DeclareLaunchArgument(
                "search_observe_duration",
                default_value="0.0",
                description="Stationary VLM observation time; zero means continuous search",
            ),
            DeclareLaunchArgument(
                "search_duration",
                default_value="60.0",
                description="Maximum bounded-search duration",
            ),
            DeclareLaunchArgument(
                "episode_timeout",
                default_value="120.0",
                description="Maximum duration of one semantic target episode",
            ),
            DeclareLaunchArgument(
                "demo5_detect_rate",
                default_value="0.2",
                description="Qwen detector request rate in Hz",
            ),
            DeclareLaunchArgument(
                "enable_color_fallback",
                default_value="true",
                description=(
                    "Allow local red-color detection when Qwen-VL is unavailable; "
                    "set false for a strict cloud-model integration test"
                ),
            ),
            DeclareLaunchArgument(
                "vlm_model",
                default_value=os.environ.get(
                    "BPX_QWEN_VL_MODEL", "qwen3-vl-32b-thinking"
                ),
                description="DashScope Qwen-VL model for Demo5 detection",
            ),
            gazebo_launch,
            inference,
            safety_shield,
            supervisor,
            random_obstacles,
            random_target,
            vision_bridge,
            demo6_bridge,
            demo4,
            demo5,
            demo6,
            task_adapter,
        ]
    )
