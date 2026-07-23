#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ======================================================================
# 常量 & 默认配置
# ======================================================================

_DEFAULT_PARAMS: Dict[str, Any] = {
    "result_file": "",
    "publish_topic": "/vla/detection/bbox",
    "publish_rate": 1.0,           # Hz
    "watch_file": True,            # 是否只在新内容时发布
    "min_publish_interval": 0.5,   # 两次发布最小间隔(秒), 防抖动
}

_LOGGER_BORDER = "=" * 60

# 默认结果文件路径 (当 result_file 参数为空时使用)
_DEFAULT_RESULT_PATH = os.path.join(
    os.environ.get(
        "DRLNAV_BASE_PATH", os.path.expanduser("~/DRL-Robot-Navigation-ROS2")
    ),
    "results",
    "vla_bridge",
    "demo5_grounding_result.json",
)


# ======================================================================
# 领域模型
# ======================================================================


@dataclass
class DetectionResult:
    """从 JSON 解析出的检测结果 (不可变语义)"""

    success: bool
    timestamp: str
    image: str
    target: str
    bbox_pixel: Dict[str, float]
    center_pixel: Dict[str, float]
    position: str
    confidence: float

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Optional["DetectionResult"]:
        """安全地从 JSON dict 构造。任何字段异常都返回 None"""
        try:
            # 核心校验: success 必须为 True
            if not data.get("success", False):
                return None

            bbox = data.get("bbox_pixel") or {}
            center = data.get("center_pixel") or {}

            return cls(
                success=True,
                timestamp=str(data.get("timestamp", "")),
                image=str(data.get("image", "")),
                target=str(data.get("target", "unknown")),
                bbox_pixel={
                    "x1": float(bbox.get("x1", 0)),
                    "y1": float(bbox.get("y1", 0)),
                    "x2": float(bbox.get("x2", 0)),
                    "y2": float(bbox.get("y2", 0)),
                },
                center_pixel={
                    "u": float(center.get("u", 0)),
                    "v": float(center.get("v", 0)),
                },
                position=str(data.get("position", "")),
                confidence=float(data.get("confidence", 0.0)),
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            return None

    def to_json_string(self) -> str:
        """序列化为 JSON 字符串 (用于 ROS2 消息)"""
        return json.dumps({
            "success": self.success,
            "timestamp": self.timestamp,
            "image": self.image,
            "target": self.target,
            "bbox_pixel": self.bbox_pixel,
            "center_pixel": self.center_pixel,
            "position": self.position,
            "confidence": self.confidence,
        })

    @property
    def summary(self) -> str:
        """一行摘要"""
        return (
            f"{self.target} "
            f"(conf={self.confidence:.3f}, "
            f"center=({self.center_pixel['u']:.0f}, {self.center_pixel['v']:.0f}), "
            f"pos={self.position})"
        )


# ======================================================================
# 节点
# ======================================================================


class Demo5Bridge(Node):
    """Demo5 结果文件 → ROS2 话题 桥接节点"""

    def __init__(self) -> None:
        super().__init__("demo5_bridge")

        # ---------- 声明 & 读取参数 ----------
        for name, default in _DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)

        _get = lambda name: self.get_parameter(name).value

        publish_topic: str = _get("publish_topic")
        publish_rate: float = _get("publish_rate")
        self._watch_file: bool = _get("watch_file")
        self._min_interval: float = _get("min_publish_interval")

        # 结果文件路径
        raw_path: str = _get("result_file")
        self._result_file = os.path.expanduser(raw_path) if raw_path else _DEFAULT_RESULT_PATH

        # ---------- 发布器 ----------
        self._publisher = self.create_publisher(String, publish_topic, 10)

        # ---------- 文件监视状态 ----------
        self._last_mtime: float = 0.0
        self._last_publish_time: float = 0.0

        # ---------- 统计 ----------
        self._stats = {
            "publish_count": 0,
            "file_not_found": 0,
            "parse_fail": 0,
            "skip_unchanged": 0,
            "skip_throttled": 0,
        }

        # ---------- 定时器 ----------
        rate = max(0.1, publish_rate)
        self._timer = self.create_timer(1.0 / rate, self._tick)

        # ---------- 启动日志 ----------
        self._log_startup()

    # -------- 日志 --------

    def _log_startup(self) -> None:
        self.get_logger().info(_LOGGER_BORDER)
        self.get_logger().info("🚀 Demo5 ROS2 桥接节点 启动")
        self.get_logger().info(f"📁 监视文件: {self._result_file}")
        self.get_logger().info(f"📤 发布话题: {self._publisher.topic_name}")
        self.get_logger().info(f"⏱  发布频率: {self._timer.timer_period_ns / 1e9:.1f}s")
        self.get_logger().info(f"🔍 监视变化: {'启用' if self._watch_file else '禁用'}")
        self._log_file_status()
        self.get_logger().info(_LOGGER_BORDER)

    def _log_file_status(self) -> None:
        if os.path.exists(self._result_file):
            size = os.path.getsize(self._result_file)
            mtime = os.path.getmtime(self._result_file)
            self.get_logger().info(
                f"✅ 文件存在: {size:,} bytes, "
                f"最后修改: {time.strftime('%H:%M:%S', time.localtime(mtime))}"
            )
        else:
            self.get_logger().warn(f"⚠️ 文件不存在, 等待生成: {self._result_file}")
            self.get_logger().warn("   请先运行: python3 demo5_grounding_api.py")

    def _log_stats_periodic(self) -> None:
        """每 20 次发布打印一次统计"""
        n = self._stats["publish_count"]
        if n > 0 and n % 20 == 0:
            fail = self._stats["parse_fail"] + self._stats["file_not_found"]
            self.get_logger().info(
                f"📊 统计: 发布={n}, 解析失败={self._stats['parse_fail']}, "
                f"文件缺失={self._stats['file_not_found']}, "
                f"跳过(未变化)={self._stats['skip_unchanged']}"
            )

    # -------- 核心逻辑 --------

    def _tick(self) -> None:
        """定时器驱动: 读取文件 → 解析 → 发布"""
        # 1) 节流: 两次发布之间至少间隔 min_interval 秒
        now = time.time()
        if now - self._last_publish_time < self._min_interval:
            self._stats["skip_throttled"] += 1
            return

        # 2) 检查文件是否存在
        if not os.path.isfile(self._result_file):
            self._stats["file_not_found"] += 1
            # 只在文件从存在变为缺失时打日志 (避免刷屏)
            if self._last_mtime > 0:
                self.get_logger().warn(f"⚠️ 文件消失: {self._result_file}")
                self._last_mtime = 0
            return

        # 3) 检查文件是否更新 (watch_file 模式)
        mtime = os.path.getmtime(self._result_file)
        if self._watch_file and mtime == self._last_mtime:
            self._stats["skip_unchanged"] += 1
            return
        self._last_mtime = mtime

        # 4) 读取 & 解析
        result = self._read_and_parse()
        if result is None:
            return

        # 5) 发布
        msg = String()
        msg.data = result.to_json_string()
        self._publisher.publish(msg)
        self._last_publish_time = time.time()
        self._stats["publish_count"] += 1

        self.get_logger().info(f"📤 发布: {result.summary}")
        self._log_stats_periodic()

    def _read_and_parse(self) -> Optional[DetectionResult]:
        """读取 JSON 文件并解析为 DetectionResult, 失败返回 None"""
        try:
            with open(self._result_file, "r", encoding="utf-8") as f:
                raw: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            self._stats["parse_fail"] += 1
            self.get_logger().warn(f"⚠️ JSON 解析失败 (文件可能正在写入): {e}")
            return None
        except PermissionError as e:
            self._stats["parse_fail"] += 1
            self.get_logger().error(f"❌ 无权限读取: {e}")
            return None
        except OSError as e:
            self._stats["parse_fail"] += 1
            self.get_logger().error(f"❌ 文件读取异常: {e}")
            return None

        result = DetectionResult.from_json(raw)
        if result is None:
            self._stats["parse_fail"] += 1
            self.get_logger().warn(
                "⚠️ 数据格式无效 (success=false 或字段缺失), 跳过发布"
            )
            return None

        return result


# ======================================================================
# Entrypoint
# ======================================================================


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)

    # 注册 Ctrl+C 优雅退出
    node = Demo5Bridge()
    signal.signal(signal.SIGINT, lambda sig, frame: node.destroy_node())

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 节点被用户中断")
    finally:
        # 打印最终统计
        s = node._stats
        node.get_logger().info(
            f"📊 最终统计: "
            f"发布={s['publish_count']}次, "
            f"解析失败={s['parse_fail']}次"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
