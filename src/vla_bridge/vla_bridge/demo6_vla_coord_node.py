#!/usr/bin/env python3
"""
Demo6: 坐标变换节点 (Gazebo Fortress 适配版)

Pipeline:
  Demo5 (实时检测) ──→ /vla/detection/bbox
                         ↓
  Gazebo ──→ depth + camera_info ──→ 像素 → 相机光学坐标
                                             └─→ TF: camera → base_link → map
                                                        └─→ /vla/target/map_pose
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import tf2_geometry_msgs  # noqa: F401  # registers PointStamped transforms
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import TransformException


# ======================================================================
# 常量
# ======================================================================

_DEFAULT_PARAMS: Dict[str, Any] = {
    "depth_topic": "/camera/depth/image_raw",
    "camera_info_topic": "/camera/camera_info",
    "bbox_topic": "/vla/detection/bbox",
    "map_pose_topic": "/vla/target/map_pose",
    "target_frame": "map",
    # 仅当深度图 header.frame_id 为空时使用此后备值。
    # Fortress 的 RGBD sensor header 可能给出 Gazebo 内部名
    # turtlebot3_waffle/base_footprint/rgbd_camera；像素反投影后的 3D 点属于光学坐标系。
    "camera_frame": "camera_depth_optical_frame",
    "publish_rate": 2.0,
    "bbox_timeout_sec": 5.0,
    "max_depth_m": 10.0,
    "min_depth_m": 0.05,
    "maximum_sync_error": 0.20,
}

_LOGGER_BORDER = "=" * 60


def ros_depth_to_numpy(msg: Image) -> Tuple[np.ndarray, str]:
    """Convert ROS depth Image to numpy without cv_bridge."""
    encoding = msg.encoding.upper()
    if encoding == "16UC1":
        dtype = np.uint16
        channels = 1
    elif encoding == "32FC1":
        dtype = np.float32
        channels = 1
    elif encoding == "32FC3":
        dtype = np.float32
        channels = 3
        encoding = "32FC1"
    else:
        raise ValueError(f"不支持的深度图编码: {msg.encoding}")

    itemsize = np.dtype(dtype).itemsize
    row_bytes = msg.width * channels * itemsize
    if msg.step < row_bytes:
        raise ValueError(f"深度图 step 异常: step={msg.step}, expected>={row_bytes}")

    data = np.frombuffer(msg.data, dtype=dtype)
    expected_items = (msg.step * msg.height) // itemsize
    if data.size < expected_items:
        raise ValueError(f"深度图数据长度不足: {data.size} < {expected_items}")

    rows = data[:expected_items].reshape((msg.height, msg.step // itemsize))
    image = rows[:, : msg.width * channels]
    if channels == 1:
        return image.reshape((msg.height, msg.width)).copy(), encoding
    return image.reshape((msg.height, msg.width, channels))[:, :, 0].copy(), "32FC1"

# ======================================================================
# 数据类型
# ======================================================================


@dataclass
class BboxData:
    """从 Demo5 JSON 解析出的检测框数据"""

    target: str
    confidence: float
    bbox_pixel: Dict[str, float]  # {"x1": .., "y1": .., "x2": .., "y2": ..}
    center_pixel: Dict[str, float]  # {"u": .., "v": ..}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["BboxData"]:
        """安全地从 JSON dict 构造，失败返回 None 而非抛异常"""
        try:
            bbox_pixel = data.get("bbox_pixel") or data.get("bbox") or {}
            center_pixel = data.get("center_pixel") or {}
            return cls(
                target=str(data.get("target", "unknown")),
                confidence=float(data.get("confidence", 0.0)),
                bbox_pixel=bbox_pixel,
                center_pixel=center_pixel,
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            return None

    @property
    def center_u(self) -> Optional[float]:
        """获取 bbox 中心点 u (像素列坐标)"""
        if "u" in self.center_pixel:
            return float(self.center_pixel["u"])
        x1 = self.bbox_pixel.get("x1")
        x2 = self.bbox_pixel.get("x2")
        if x1 is not None and x2 is not None:
            return (float(x1) + float(x2)) / 2.0
        return None

    @property
    def center_v(self) -> Optional[float]:
        """获取 bbox 中心点 v (像素行坐标)"""
        if "v" in self.center_pixel:
            return float(self.center_pixel["v"])
        y1 = self.bbox_pixel.get("y1")
        y2 = self.bbox_pixel.get("y2")
        if y1 is not None and y2 is not None:
            return (float(y1) + float(y2)) / 2.0
        return None

    @property
    def is_valid(self) -> bool:
        """检测框是否包含有效的像素坐标"""
        return self.center_u is not None and self.center_v is not None


@dataclass
class CameraModel:
    """针孔相机模型 (简化版)"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, info: CameraInfo) -> "CameraModel":
        return cls(
            fx=info.k[0],
            fy=info.k[4],
            cx=info.k[2],
            cy=info.k[5],
            width=info.width,
            height=info.height,
        )

    def pixel_to_camera(self, u: float, v: float, depth: float) -> Tuple[float, float, float]:
        """像素坐标 → 相机坐标系 3D 点

        公式:
            Xc = (u - cx) * depth / fx
            Yc = (v - cy) * depth / fy
            Zc = depth
        """
        Xc = (u - self.cx) * depth / self.fx
        Yc = (v - self.cy) * depth / self.fy
        Zc = depth
        return Xc, Yc, Zc

    def is_in_bounds(self, u: int, v: int) -> bool:
        """像素是否在图像范围内"""
        return 0 <= u < self.width and 0 <= v < self.height


# ======================================================================
# 节点
# ======================================================================


class CoordTransformNode(Node):
    """坐标变换节点: Bbox 像素 → 地图坐标 (TF)"""

    def __init__(self) -> None:
        super().__init__("coord_transform_node")

        # ---------- 声明 & 读取参数 ----------
        for name, default in _DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)

        _get = lambda name: self.get_parameter(name).value  # noqa: E731
        self.depth_topic: str = _get("depth_topic")
        self.camera_info_topic: str = _get("camera_info_topic")
        self.bbox_topic: str = _get("bbox_topic")
        self.map_pose_topic: str = _get("map_pose_topic")
        self.target_frame: str = _get("target_frame")
        self.camera_frame: str = _get("camera_frame")
        self.publish_rate: float = _get("publish_rate")
        self.bbox_timeout_sec: float = _get("bbox_timeout_sec")
        self.max_depth_m: float = _get("max_depth_m")
        self.min_depth_m: float = _get("min_depth_m")
        self.maximum_sync_error: float = _get("maximum_sync_error")

        # ---------- 基础设施 ----------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---------- 内部状态 ----------
        self._latest_depth: Optional[Image] = None
        self._depth_history = deque(maxlen=400)
        self._camera_model: Optional[CameraModel] = None
        self._latest_bbox: Optional[BboxData] = None
        self._bbox_timestamp: Optional[float] = None
        self._bbox_source_time: Optional[float] = None
        self._reported_frame_remaps: set[Tuple[str, str]] = set()

        # ---------- 统计 ----------
        self._stats = {"tf_success": 0, "tf_fail": 0, "depth_invalid": 0, "total_processed": 0}

        # ---------- QoS ----------
        self._best_effort_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ---------- 订阅器 ----------
        self._sub_depth = self.create_subscription(
            Image, self.depth_topic, self._on_depth, self._best_effort_qos
        )
        self._sub_camera_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, self._best_effort_qos
        )
        self._sub_bbox = self.create_subscription(
            String, self.bbox_topic, self._on_bbox, 10
        )
        self._sub_semantic_reset = self.create_subscription(
            Bool,
            "/vla_navigation/semantic_reset",
            self._on_semantic_reset,
            10,
        )

        # ---------- 发布器 ----------
        self._pub_map_pose = self.create_publisher(PoseStamped, self.map_pose_topic, 10)

        # ---------- 定时器 ----------
        rate = max(0.1, self.publish_rate)
        self._timer = self.create_timer(1.0 / rate, self._process)

        # ---------- 启动日志 ----------
        self._log_startup()

    # -------- 日志辅助 --------

    def _log_startup(self) -> None:
        self.get_logger().info(_LOGGER_BORDER)
        self.get_logger().info("✅ Demo6: 坐标变换节点启动 (Fortress 版)")
        self.get_logger().info(f"📌 深度图:        {self.depth_topic}")
        self.get_logger().info(f"📌 相机内参:      {self.camera_info_topic}")
        self.get_logger().info(f"📌 检测框:        {self.bbox_topic}")
        self.get_logger().info(f"📌 发布地图坐标:   {self.map_pose_topic}")
        self.get_logger().info(f"📌 目标坐标系:    {self.target_frame}")
        self.get_logger().info(f"📌 相机坐标系后备值: {self.camera_frame}")
        self.get_logger().info(f"📌 发布频率:      {self.publish_rate} Hz")
        self.get_logger().info(_LOGGER_BORDER)
        self.get_logger().warn(
            f"请确保 TF 链可用: camera_depth_optical_frame -> camera_link -> "
            f"base_link/base_footprint -> odom -> {self.target_frame}"
        )

    def _log_stats_periodic(self) -> None:
        """每 50 次处理打印一次统计摘要"""
        total = self._stats["total_processed"]
        if total > 0 and total % 50 == 0:
            tf_ok = self._stats["tf_success"]
            tf_fail = self._stats["tf_fail"]
            depth_invalid = self._stats["depth_invalid"]
            tf_rate = tf_ok / max(1, tf_ok + tf_fail) * 100
            self.get_logger().info(
                f"📊 统计: 处理={total}, TF成功率={tf_rate:.0f}%, "
                f"无效深度={depth_invalid}"
            )

    # -------- 回调函数 --------

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = msg
        self._depth_history.append(msg)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._camera_model is None:
            self._camera_model = CameraModel.from_camera_info(msg)
            self.get_logger().info(
                f"📷 相机: {self._camera_model.width}x{self._camera_model.height}, "
                f"fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}"
            )

    def _on_bbox(self, msg: String) -> None:
        try:
            raw: Dict[str, Any] = json.loads(msg.data)
            if not isinstance(raw, dict):
                raise ValueError("JSON 根节点必须是对象")
        except (json.JSONDecodeError, ValueError) as e:
            self.get_logger().warn(f"⚠️ JSON 解析失败: {e}")
            return

        # Demo5 的阴性结果也使用固定 schema。阴性或非 red-box 消息会
        # 立即清除旧框，避免 Demo6 继续发布过期坐标。
        target = str(raw.get("target", "")).strip().lower()
        if not bool(raw.get("success", False)) or target not in (
            "red box",
            "red cube",
            "red block",
        ):
            self._latest_bbox = None
            self._bbox_timestamp = None
            self._bbox_source_time = None
            return

        bbox = BboxData.from_dict(raw)
        if bbox is None or not bbox.is_valid:
            self.get_logger().warn("⚠️ 检测框数据格式无效，跳过")
            return

        self._latest_bbox = bbox
        self._bbox_timestamp = time.time()
        stamp = raw.get("source_stamp") or {}
        try:
            self._bbox_source_time = (
                float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
            )
        except (KeyError, TypeError, ValueError):
            self._bbox_source_time = None
        self.get_logger().info(
            f"📦 {bbox.target} 置信度={bbox.confidence:.3f} "
            f"中心=({bbox.center_u:.0f}, {bbox.center_v:.0f})"
        )

    def _on_semantic_reset(self, msg: Bool) -> None:
        if not msg.data:
            return
        self._latest_bbox = None
        self._bbox_timestamp = None
        self._bbox_source_time = None
        self.get_logger().info("🆕 新语义目标：已清除上一任务检测框")

    @staticmethod
    def _stamp_seconds(stamp: Any) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _synchronized_depth(self) -> Optional[Image]:
        if self._bbox_source_time is None:
            return self._latest_depth
        if not self._depth_history:
            return None
        depth = min(
            self._depth_history,
            key=lambda item: abs(
                self._stamp_seconds(item.header.stamp) - self._bbox_source_time
            ),
        )
        error = abs(
            self._stamp_seconds(depth.header.stamp) - self._bbox_source_time
        )
        if error > self.maximum_sync_error:
            self.get_logger().warn(
                f"⚠️ 检测框与深度图不同步: {error:.3f}s",
                throttle_duration_sec=5.0,
            )
            return None
        return depth

    # -------- 核心处理 --------

    def _process(self) -> None:
        """主处理 (定时器驱动)"""

        # 1) 前置条件检查
        if not self._preconditions_met():
            return

        bbox = self._latest_bbox  # 类型收窄
        depth_msg = self._synchronized_depth()
        camera = self._camera_model
        if depth_msg is None:
            return

        # 2) 深度图 → numpy
        depth_map, depth_encoding = self._decode_depth(depth_msg)
        if depth_map is None:
            return

        # 3) 读取中心点深度
        u, v = bbox.center_u, bbox.center_v  # type: ignore[union-attr]
        u_int, v_int = int(round(u)), int(round(v))

        if not camera.is_in_bounds(u_int, v_int):
            self.get_logger().warn(f"⚠️ 像素越界 ({u_int}, {v_int}) / ({camera.width}, {camera.height})")
            return

        depth_m = self._extract_depth(depth_map, v_int, u_int, depth_encoding)
        if depth_m is None:
            self.get_logger().warn(
                f"⚠️ bbox 中心附近没有有效深度: center=({u_int}, {v_int})"
            )
            self._stats["depth_invalid"] += 1
            return

        # Gazebo 无效深度通常是 NaN、Inf 或 0。在参与投影前统一过滤。
        if not math.isfinite(depth_m):
            self.get_logger().warn(f"⚠️ 深度值无效 (NaN/Inf): {depth_m}")
            self._stats["depth_invalid"] += 1
            return

        if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
            self.get_logger().warn(f"⚠️ 深度超出范围: {depth_m:.3f}m")
            self._stats["depth_invalid"] += 1
            return

        # 4) 像素 → 相机坐标
        Xc, Yc, Zc = camera.pixel_to_camera(float(u), float(v), depth_m)

        # 5) TF 会沿树自动完成 camera optical -> base_link -> map。
        # 使用深度图的 frame_id/时间戳，避免手写重复中间变换。
        source_frame = self._normalize_source_frame(depth_msg.header.frame_id)
        map_pose = self._transform_to_map(
            Xc, Yc, Zc, source_frame, depth_msg.header.stamp
        )
        if map_pose is None:
            return

        # 6) 发布 PoseStamped
        self._publish_pose(map_pose)
        self._stats["total_processed"] += 1

        self.get_logger().info(
            f"🎯 {bbox.target} → 地图 ({map_pose['x']:.3f}, {map_pose['y']:.3f}, {map_pose['z']:.3f})m "
            f"[深度={depth_m:.3f}m]"
        )
        self._log_stats_periodic()

    def _preconditions_met(self) -> bool:
        """检查所有必要数据是否就绪"""
        if self._latest_depth is None:
            return False
        if self._camera_model is None:
            return False
        if self._latest_bbox is None:
            return False
        if self._bbox_timestamp is None:
            return False
        if time.time() - self._bbox_timestamp > self.bbox_timeout_sec:
            return False
        return True

    # -------- 深度图处理 --------

    def _decode_depth(self, msg: Image) -> Tuple[Optional[np.ndarray], str]:
        """将 ROS Image 转为 numpy 深度图, 返回 (array, encoding) 或 (None, '')"""
        encoding = msg.encoding  # 先看消息自带的编码

        try:
            return ros_depth_to_numpy(msg)
        except Exception as exc:
            self.get_logger().warn(f"⚠️ 深度图解码失败 ({encoding}): {exc}")
            return None, ""

    def _extract_depth(
        self, depth_map: np.ndarray, row: int, col: int, encoding: str
    ) -> Optional[float]:
        """从 bbox 中心附近提取有效深度值 (米)。

        Gazebo 深度图在物体边缘/遮挡处经常出现 NaN、Inf 或 0。
        只取中心单像素会让 Demo6 误判失败，所以这里从中心向外扩展小窗口，
        取合法深度的中位数，既实时又比单点稳定。
        """
        if row < 0 or col < 0 or row >= depth_map.shape[0] or col >= depth_map.shape[1]:
            return None

        for radius in (0, 2, 4, 8, 12, 20):
            r0 = max(0, row - radius)
            r1 = min(depth_map.shape[0], row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(depth_map.shape[1], col + radius + 1)

            values = depth_map[r0:r1, c0:c1].astype(np.float32, copy=False)

            # 16UC1: 毫米 → 米
            if encoding == "16UC1":
                values_m = values / 1000.0
            elif encoding == "32FC1":
                values_m = values
            else:
                values_m = values / 1000.0 if np.nanmax(values) > 100.0 else values

            valid = values_m[
                np.isfinite(values_m)
                & (values_m >= self.min_depth_m)
                & (values_m <= self.max_depth_m)
            ]
            if valid.size > 0:
                return float(np.median(valid))

        return None

    def _normalize_source_frame(self, frame_id: str) -> str:
        """把 Gazebo sensor 内部 frame 名规范成 TF 树中真实存在的 frame。"""
        raw_frame = (frame_id or "").strip()
        if not raw_frame:
            return self.camera_frame

        normalized = raw_frame
        if raw_frame == "rgbd_camera" or raw_frame.endswith("/rgbd_camera"):
            normalized = self.camera_frame

        if normalized != raw_frame:
            key = (raw_frame, normalized)
            if key not in self._reported_frame_remaps:
                self._reported_frame_remaps.add(key)
                self.get_logger().info(
                    f"🔁 深度图 frame 映射: {raw_frame} -> {normalized}"
                )

        return normalized

    # -------- TF 变换 --------

    def _transform_to_map(
        self,
        Xc: float,
        Yc: float,
        Zc: float,
        source_frame: str,
        stamp: Any,
    ) -> Optional[Dict[str, float]]:
        """相机光学坐标 → TF 树(含本体) → 地图坐标。"""
        point_camera = PointStamped()
        point_camera.header.frame_id = source_frame
        point_camera.header.stamp = stamp
        point_camera.point.x = Xc
        point_camera.point.y = Yc
        point_camera.point.z = Zc

        try:
            point_map = self._tf_buffer.transform(point_camera, self.target_frame)
        except TransformException as e:
            self._stats["tf_fail"] += 1
            if self._stats["tf_fail"] % 10 == 0:
                self.get_logger().warn(
                    f"⚠️ TF 变换失败 {source_frame} -> "
                    f"{self.target_frame} (累计 {self._stats['tf_fail']} 次): {e}"
                )
            return None

        self._stats["tf_success"] += 1
        return {
            "x": point_map.point.x,
            "y": point_map.point.y,
            "z": point_map.point.z,
        }

    # -------- 发布 --------

    def _publish_pose(self, map_data: Dict[str, float]) -> None:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.target_frame
        pose.pose.position.x = map_data["x"]
        pose.pose.position.y = map_data["y"]
        pose.pose.position.z = map_data["z"]
        pose.pose.orientation.w = 1.0
        self._pub_map_pose.publish(pose)


# ======================================================================
# Entrypoint
# ======================================================================


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = CoordTransformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("🛑 节点被用户中断")
    finally:
        try:
            node.destroy_node()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except (Exception, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
