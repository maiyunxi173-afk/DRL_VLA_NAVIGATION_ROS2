#!/usr/bin/env python3
"""Demo4: real-time VLM target planner for Demo5/Demo6.

Input:
  * /camera/color/image_raw (sensor_msgs/Image)
  * /vla/user_command (std_msgs/String, optional target hint)
Output:
  * <DRLNAV_BASE_PATH>/results/vla_bridge/demo4_vlm_result.json
  * /vla/demo4/result (std_msgs/String, same JSON)

Demo5 reads the JSON file and uses target_object for open-vocabulary detection.
By default Demo4 performs automatic scene discovery instead of relying on a
pre-filled target such as "red box".
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rclpy
from dashscope import MultiModalConversation
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


DEFAULT_RESULT_DIR = os.path.join(
    os.environ.get(
        "DRLNAV_BASE_PATH", os.path.expanduser("~/DRL-Robot-Navigation-ROS2")
    ),
    "results",
    "vla_bridge",
)
DEFAULT_RESULT_FILE = os.path.join(DEFAULT_RESULT_DIR, "demo4_vlm_result.json")
DEFAULT_FRAME_FILE = os.path.join(DEFAULT_RESULT_DIR, "demo4_latest_frame.jpg")
DEFAULT_FREE_QUOTA_VISION_MODEL = "qwen3-vl-32b-thinking"
FREE_QUOTA_VISION_MODELS = (
    "qwen3-vl-32b-thinking",
    "qwen3-vl-235b-a22b-thinking",
)
OPTIONAL_VISION_MODELS = (
    "qwen3-vl-plus",
    "qwen3-vl-plus-2025-09-23",
    "qwen3-vl-flash",
    "qwen3-vl-32b-instruct",
    "qwen3-vl-235b-a22b-instruct",
    "qwen-vl-plus",
    "qwen-vl-plus-latest",
    "qwen-vl-plus-2025-07-10",
    "qwen-vl-plus-2025-08-15",
    "qwen-vl-max",
    "qwen-vl-max-latest",
    "qwen-vl-max-2025-04-08",
    "qwen-vl-max-2025-08-13",
    "qwen2.5-vl-3b-instruct",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen2.5-vl-72b-instruct",
    "qwen3.7-plus",
)
DEFAULT_VISION_MODEL = os.getenv(
    "BPX_QWEN_VL_MODEL", DEFAULT_FREE_QUOTA_VISION_MODEL
)
ALLOWED_DASHSCOPE_MODELS = set(FREE_QUOTA_VISION_MODELS + OPTIONAL_VISION_MODELS)


def _validate_dashscope_model(model: str) -> str:
    model = str(model).strip()
    if model not in ALLOWED_DASHSCOPE_MODELS:
        free_models = ", ".join(FREE_QUOTA_VISION_MODELS)
        optional_models = ", ".join(OPTIONAL_VISION_MODELS)
        raise RuntimeError(
            "Demo4 需要支持图像输入的 Qwen-VL/多模态模型。"
            f"免费额度优先模型: {free_models}；其他可选视觉模型: {optional_models}；"
            f"当前配置: {model or '<空>'}"
        )
    return model


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    """Convert common ROS color Image encodings to OpenCV BGR without cv_bridge."""
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }
    encoding = msg.encoding.lower()
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"不支持的彩色图编码: {msg.encoding}")

    if msg.height <= 0 or msg.width <= 0:
        raise ValueError("图像宽高无效")

    row_bytes = msg.width * channels
    if msg.step < row_bytes:
        raise ValueError(f"图像 step 异常: step={msg.step}, expected>={row_bytes}")

    data = np.frombuffer(msg.data, dtype=np.uint8)
    expected = msg.step * msg.height
    if data.size < expected:
        raise ValueError(f"图像数据长度不足: {data.size} < {expected}")

    image = data[:expected].reshape((msg.height, msg.step))
    image = image[:, :row_bytes].reshape((msg.height, msg.width, channels))

    if encoding == "bgr8":
        return image.copy()
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"不支持的彩色图编码: {msg.encoding}")


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse a strict JSON response, or recover the first JSON object in text."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None

    try:
        data = json.loads(match.group())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_response_text(response: Any) -> str:
    status_code = getattr(response, "status_code", None)
    code = getattr(response, "code", "")
    message = getattr(response, "message", "")
    request_id = getattr(response, "request_id", "")
    output = getattr(response, "output", None)
    if output is None:
        raise RuntimeError(
            "DashScope 响应为空: "
            f"status_code={status_code}, code={code}, message={message}, request_id={request_id}"
        )

    choices = getattr(output, "choices", None)
    if not choices:
        raise RuntimeError(
            "DashScope 响应缺少 choices: "
            f"status_code={status_code}, code={code}, message={message}, request_id={request_id}"
        )

    content = choices[0].message.content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".demo4_", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _build_prompt(user_command: str = "") -> str:
    command = user_command.strip()
    command_block = (
        f"用户给出的任务提示: {command}\n"
        if command
        else "固定任务目标是 red box。\n"
    )

    return f"""固定任务: 寻找 red box。必须看当前图像，blue box 不算成功。
{command_block}
只输出 JSON:
{{"scene_objects":[],"target_visible":false,"action":"search","reason":""}}
规则: scene_objects 用英文短语。只有明确看到红色盒子/红色立方体时 target_visible=true 且 action="approach"，否则 target_visible=false 且 action="search"。不要输出 target_object。
"""


def _evaluate_red_box(parsed: Dict[str, Any], scene_objects: List[str]) -> Dict[str, Any]:
    """Accept only an explicit red box observation; blue or red surfaces never pass."""
    value = str(parsed.get("target_object", "")).strip().lower()
    objects = [item.strip().lower() for item in scene_objects]
    all_terms = [value] + objects

    if any(term in ("red box", "red cube", "red block") for term in all_terms):
        return {
            "target_visible": True,
            "confidence_note": "explicit red box/cube/block observation",
        }
    return {
        "target_visible": False,
        "confidence_note": "red box is not explicitly visible",
    }


def _local_red_box_visible(frame: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    height, width = frame.shape[:2]
    min_area = max(100.0, width * height * 0.0005)
    area = float(cv2.contourArea(max(contours, key=cv2.contourArea)))
    return area >= min_area


def _build_local_result(
    frame_file: str,
    frame_id: str,
    stamp: str,
    command: str,
    target_visible: bool,
    reason: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": os.path.basename(frame_file),
        "frame_id": frame_id,
        "image_stamp": stamp,
        "command": command,
        "mission_target": "red box",
        "scene_objects": ["red box"] if target_visible else [],
        "target_object": "red box",
        "target_visible": target_visible,
        "action": "approach" if target_visible else "search",
        "needs_better_view": not target_visible,
        "confidence_note": "local red color observation",
        "reason": reason,
        "error": "",
    }


def _choose_fallback_target(scene_objects: List[str]) -> str:
    """Pick a target when the VLM lists objects but omits target_object."""
    if not scene_objects:
        return ""

    explicit_box_terms = {"red box", "red cube", "red block", "blue box", "blue cube"}
    for item in scene_objects:
        if item.strip().lower() in explicit_box_terms:
            return item.strip()

    weak_surface_words = ("wall", "surface", "floor", "ground", "edge")
    for item in scene_objects:
        value = item.strip()
        if value and not any(word in value.lower() for word in weak_surface_words):
            return value

    return scene_objects[0].strip()


def analyze_scene(
    image_path: str,
    user_command: str = "",
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = 512,
    result_file: str = DEFAULT_RESULT_FILE,
) -> Dict[str, Any]:
    """Analyze one image and write the Demo5-compatible result JSON."""
    image_path = os.path.expanduser(image_path)
    api_key = (api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY，无法调用 Demo4 VLM")
    model = _validate_dashscope_model(model)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": _build_prompt(user_command)},
            ],
        }
    ]

    response = MultiModalConversation.call(
        model=model,
        api_key=api_key,
        messages=messages,
        max_tokens=max(64, min(int(max_tokens), 2048)),
        enable_thinking=False,
    )

    parsed = parse_json(_extract_response_text(response))
    if not parsed:
        raise RuntimeError("VLM 返回内容不是有效 JSON")

    scene_objects_raw = parsed.get("scene_objects", [])
    if isinstance(scene_objects_raw, list):
        scene_objects = [str(item).strip() for item in scene_objects_raw if str(item).strip()]
    elif scene_objects_raw:
        scene_objects = [str(scene_objects_raw).strip()]
    else:
        scene_objects = []

    evaluated = _evaluate_red_box(parsed, scene_objects)
    target_visible = bool(evaluated["target_visible"])
    reason = str(parsed.get("reason", "")).strip()

    output = {
        "schema_version": "1.0",
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": os.path.basename(image_path),
        "frame_id": "",
        "image_stamp": "",
        "command": user_command,
        "mission_target": "red box",
        "scene_objects": scene_objects,
        "target_object": "red box",
        "target_visible": target_visible,
        "action": "approach" if target_visible else "search",
        "needs_better_view": not target_visible,
        "confidence_note": str(evaluated["confidence_note"]),
        "reason": reason,
        "error": "",
    }
    _atomic_write_json(result_file, output)
    return output


class Demo4RealtimeVlmNode(Node):
    """Realtime latest-frame VLM node feeding Demo5 target JSON."""

    def __init__(self) -> None:
        super().__init__("demo4_realtime_vlm")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("command_topic", "/vla/user_command")
        self.declare_parameter("publish_topic", "/vla/demo4/result")
        self.declare_parameter("result_file", DEFAULT_RESULT_FILE)
        self.declare_parameter("frame_file", DEFAULT_FRAME_FILE)
        self.declare_parameter("default_command", "")
        self.declare_parameter("model", DEFAULT_VISION_MODEL)
        self.declare_parameter("max_tokens", 512)
        self.declare_parameter("analyze_rate", 0.2)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("cloud_image_max_width", 640)
        self.declare_parameter("cloud_image_max_height", 480)
        self.declare_parameter("run_on_command", True)
        self.declare_parameter("auto_scene_discovery", True)
        self.declare_parameter("service_error_cooldown_sec", 60.0)

        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        self._command_topic = str(self.get_parameter("command_topic").value)
        publish_topic = str(self.get_parameter("publish_topic").value)
        self._result_file = os.path.expanduser(str(self.get_parameter("result_file").value))
        self._frame_file = os.path.expanduser(str(self.get_parameter("frame_file").value))
        self._command = str(self.get_parameter("default_command").value).strip()
        self._model = _validate_dashscope_model(str(self.get_parameter("model").value))
        self._max_tokens = int(self.get_parameter("max_tokens").value)
        rate = max(0.01, float(self.get_parameter("analyze_rate").value))
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self._cloud_image_max_width = int(
            self.get_parameter("cloud_image_max_width").value
        )
        self._cloud_image_max_height = int(
            self.get_parameter("cloud_image_max_height").value
        )
        self._run_on_command = bool(self.get_parameter("run_on_command").value)
        self._auto_scene_discovery = bool(
            self.get_parameter("auto_scene_discovery").value
        )
        self._service_error_cooldown = max(
            5.0, float(self.get_parameter("service_error_cooldown_sec").value)
        )

        self._api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = ""
        self._latest_stamp = ""
        self._busy = False
        self._last_command_analyzed = ""
        self._retry_not_before = 0.0

        self._pub = self.create_publisher(String, publish_topic, 10)
        self._image_sub = self.create_subscription(
            Image, self._rgb_topic, self._on_image, qos_profile_sensor_data
        )
        self._command_sub = self.create_subscription(
            String, self._command_topic, self._on_command, 10
        )
        self._timer = self.create_timer(1.0 / rate, self._tick)

        if not self._api_key:
            self.get_logger().error("未设置 DASHSCOPE_API_KEY，Demo4 不会调用 VLM")

        self.get_logger().info(
            f"Demo4 实时启动: image={self._rgb_topic}, command={self._command_topic}, "
            f"result_file={self._result_file}, rate={rate:.2f}Hz"
        )

    def _on_image(self, msg: Image) -> None:
        try:
            frame = ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"图像转换失败: {exc}", throttle_duration_sec=5.0)
            return

        with self._frame_lock:
            self._latest_frame = np.asarray(frame).copy()
            self._latest_frame_id = msg.header.frame_id
            self._latest_stamp = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip()
        if not command:
            return
        self._command = command
        if self._run_on_command:
            self._last_command_analyzed = ""
        self.get_logger().info(f"收到新任务: {self._command}")

    def _tick(self) -> None:
        if self._busy or not self._api_key:
            return
        if time.monotonic() < self._retry_not_before:
            return
        if (
            self._run_on_command
            and not self._auto_scene_discovery
            and self._command == self._last_command_analyzed
        ):
            return

        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            frame_id = self._latest_frame_id
            stamp = self._latest_stamp

        if frame is None:
            self.get_logger().warn("等待相机图像...", throttle_duration_sec=5.0)
            return

        self._busy = True
        try:
            self._save_frame(frame)
            if _local_red_box_visible(frame):
                result = _build_local_result(
                    self._frame_file,
                    frame_id,
                    stamp,
                    self._command,
                    True,
                    "local red box detector confirmed target",
                )
                _atomic_write_json(self._result_file, result)
            else:
                result = analyze_scene(
                    self._frame_file,
                    self._command,
                    api_key=self._api_key,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    result_file=self._result_file,
                )
                result["frame_id"] = frame_id
                result["image_stamp"] = stamp
                _atomic_write_json(self._result_file, result)

            msg = String()
            msg.data = json.dumps(result, ensure_ascii=False)
            self._pub.publish(msg)
            self._last_command_analyzed = self._command
            self.get_logger().info(
                f"Demo4 自动观察: objects={result.get('scene_objects', [])} "
                f"target={result['target_object']} action={result['action']} "
                f"needs_better_view={result.get('needs_better_view', False)}"
            )
        except Exception as exc:
            error_text = str(exc)
            service_rejected = any(
                marker in error_text.lower()
                for marker in (
                    "arrearage",
                    "allocationquota",
                    "free quota",
                    "access denied",
                    "status_code=401",
                    "status_code=403",
                )
            )
            if service_rejected:
                self._retry_not_before = (
                    time.monotonic() + self._service_error_cooldown
                )
            # 即使云端偶发返回非 JSON，也发布完全相同的数据结构。
            # target_object 是任务目标；target_visible=false 才表示本帧未确认目标。
            failure = {
                "schema_version": "1.0",
                "success": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image": os.path.basename(self._frame_file),
                "frame_id": frame_id,
                "image_stamp": stamp,
                "command": self._command,
                "mission_target": "red box",
                "scene_objects": [],
                "target_object": "red box",
                "target_visible": False,
                "action": "hold" if service_rejected else "search",
                "needs_better_view": True,
                "confidence_note": "VLM response could not be validated",
                "reason": "service_error" if service_rejected else "inference_error",
                "error": error_text,
            }
            _atomic_write_json(self._result_file, failure)
            msg = String()
            msg.data = json.dumps(failure, ensure_ascii=False)
            self._pub.publish(msg)
            if service_rejected:
                self.get_logger().error(
                    "Demo4 云端服务拒绝请求，机器人不应据此旋转；"
                    f"{self._service_error_cooldown:.0f} 秒后重试: {exc}"
                )
            else:
                self.get_logger().error(f"Demo4 VLM 分析失败，进入搜索状态: {exc}")
        finally:
            self._busy = False

    def _save_frame(self, frame: np.ndarray) -> None:
        Path(self._frame_file).parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            self._cloud_image_max_width / max(1, width),
            self._cloud_image_max_height / max(1, height),
        )
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok = cv2.imwrite(
            self._frame_file,
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"保存相机帧失败: {self._frame_file}")


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = Demo4RealtimeVlmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户中断")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
