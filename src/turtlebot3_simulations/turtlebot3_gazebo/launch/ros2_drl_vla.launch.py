import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    model = os.environ.get("TURTLEBOT3_MODEL", "waffle")
    if model != "waffle":
        raise RuntimeError("The basic VLA camera test currently requires waffle")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    pause = LaunchConfiguration("pause", default="false")
    gazebo_share = get_package_share_directory("turtlebot3_gazebo")
    gazebo_ros_share = get_package_share_directory("gazebo_ros")
    world = os.path.join(
        gazebo_share, "worlds", "turtlebot3_drl_vla", "waffle.model"
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo_ros_share, "launch", "gzserver.launch.py")
                ),
                launch_arguments={"world": world, "pause": pause}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(gazebo_ros_share, "launch", "gzclient.launch.py")
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        gazebo_share, "launch", "robot_state_publisher.launch.py"
                    )
                ),
                launch_arguments={"use_sim_time": use_sim_time}.items(),
            ),
        ]
    )
