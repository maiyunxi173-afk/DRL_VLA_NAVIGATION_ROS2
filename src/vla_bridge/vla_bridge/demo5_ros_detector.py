#!/usr/bin/env python3
"""Demo5: Qwen-VL real-time bbox detector for Demo4 -> Demo6.

Input:
  * /camera/color/image_raw (sensor_msgs/Image)
  * <DRLNAV_BASE_PATH>/results/vla_bridge/demo4_vlm_result.json (target_object)

Output:
  * /vla/detection/bbox (std_msgs/String JSON)

The output schema is kept compatible with Demo6:
  bbox_pixel, center_pixel, confidence, target, image_width, image_height.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from dashscope import MultiModalConversation
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


DEFAULT_DEMO4_RESULT = os.path.join(
    os.environ.get(
        "DRLNAV_BASE_PATH", os.path.expanduser("~/DRL-Robot-Navigation-ROS2")
    ),
    "results",
    "vla_bridge",
    "demo4_vlm_result.json",
)
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


def validate_dashscope_model(model: str) -> str:
    model = str(model).strip()
    if model not in ALLOWED_DASHSCOPE_MODELS:
        free_models = ", ".join(FREE_QUOTA_VISION_MODELS)
        optional_models = ", ".join(OPTIONAL_VISION_MODELS)
        raise RuntimeError(
            "Demo5 需要支持图像输入的 Qwen-VL/多模态模型。"
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
    """Parse strict JSON or recover the first JSON object from model text."""
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


def extract_response_text(response: Any) -> str:
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


def build_bbox_prompt(target: str, width: int, height: int) -> str:
    return f"""检测图像中的 "{target}"。只输出 JSON。
坐标范围: x=0..{width - 1}, y=0..{height - 1}。
看到目标: {{"success":true,"target":"{target}","bbox_pixel":{{"x1":0,"y1":0,"x2":0,"y2":0}},"confidence":0.0}}
未看到目标: {{"success":false,"target":"{target}","bbox_pixel":null,"confidence":0.0}}
只接受 red box；blue box 不算目标。
"""


class Demo5QwenDetector(Node):
    """Latest-frame Qwen-VL detector compatible with Demo4 and Demo6."""

    def __init__(self) -> None:
        super().__init__("demo5_ros_detector")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("publish_topic", "/vla/detection/bbox")
        self.declare_parameter("demo4_result_file", DEFAULT_DEMO4_RESULT)
        self.declare_parameter("target_object", "red box")
        self.declare_parameter("confidence_threshold", 0.25)
        # Qwen-VL 是云端 VLM，默认 0.2Hz = 5 秒一次，和 Demo4 节奏一致。
        self.declare_parameter("detect_rate", 0.2)
        self.declare_parameter("local_detect_rate", 2.0)
        self.declare_parameter("vlm_model", DEFAULT_VISION_MODEL)
        self.declare_parameter("max_tokens", 384)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("cloud_image_max_width", 640)
        self.declare_parameter("cloud_image_max_height", 480)
        # Fortress 测试场景中的目标是高饱和纯红色箱体。云端服务异常时，
        # 使用当前相机画面做本地颜色分割，仍属于真实视觉检测，并非预置 bbox。
        self.declare_parameter("enable_color_fallback", True)

        self._rgb_topic = str(self.get_parameter("rgb_topic").value)
        publish_topic = str(self.get_parameter("publish_topic").value)
        self._demo4_file = os.path.expanduser(
            str(self.get_parameter("demo4_result_file").value)
        )
        self._fallback_target = str(self.get_parameter("target_object").value).strip()
        self._threshold = float(self.get_parameter("confidence_threshold").value)
        detect_rate = max(0.1, float(self.get_parameter("detect_rate").value))
        self._local_detect_period = 1.0 / max(
            0.1, float(self.get_parameter("local_detect_rate").value)
        )
        self._vlm_model = validate_dashscope_model(
            str(self.get_parameter("vlm_model").value).strip()
        )
        self._max_tokens = int(self.get_parameter("max_tokens").value)
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self._cloud_image_max_width = int(
            self.get_parameter("cloud_image_max_width").value
        )
        self._cloud_image_max_height = int(
            self.get_parameter("cloud_image_max_height").value
        )
        self._enable_color_fallback = bool(
            self.get_parameter("enable_color_fallback").value
        )
        self._api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = ""
        self._latest_source_stamp = {"sec": 0, "nanosec": 0}
        self._future: Optional[Future] = None
        self._future_source_stamp = {"sec": 0, "nanosec": 0}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen_detector")
        # 本项目的任务目标是固定的 red box。Demo4 负责报告它在当前画面
        # 是否可见，但不能把 blue box 改成下游检测目标。
        self._target = "red box"
        self._demo4_mtime: Optional[float] = None
        self._warned_no_target = False
        self._warned_bad_demo4 = False
        self._warned_no_key = False
        self._warned_vlm_failure = False
        self._frame_number = 0
        self._last_local_check_at = 0.0
        self._last_local_success_at = 0.0

        if not self._api_key:
            self.get_logger().error("未设置 DASHSCOPE_API_KEY，Demo5 Qwen 检测无法调用云端模型")

        self._publisher = self.create_publisher(String, publish_topic, 10)
        self._subscription = self.create_subscription(
            Image, self._rgb_topic, self._on_image, qos_profile_sensor_data
        )
        self._detect_timer = self.create_timer(1.0 / detect_rate, self._schedule_detection)
        # Poll separately so a fast local detector is not held until the next
        # 5-second cloud-inference tick.  The source image timestamp remains in
        # the result for downstream LaserScan synchronization.
        self._result_timer = self.create_timer(0.05, self._publish_completed)

        self._refresh_target(force=True)
        self.get_logger().info(
            f"Demo5 Qwen 启动: {self._rgb_topic} -> {publish_topic}, "
            f"model={self._vlm_model}, 检测频率={detect_rate:.2f}Hz, "
            f"target={self._target or '<等待 Demo4>'}"
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
            self._latest_source_stamp = {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            }

        now = time.monotonic()
        if (
            self._enable_color_fallback
            and now - self._last_local_check_at >= self._local_detect_period
        ):
            self._last_local_check_at = now
            self._frame_number += 1
            result = self._detect_colored_box(
                frame, self._target, msg.header.frame_id, self._frame_number
            )
            if result is not None:
                result["detector"] = "local_color_primary"
                result["reason"] = "red box detected by local color vision"
                self._last_local_success_at = now
                self._publish_result(result, self._latest_source_stamp)

    def _refresh_target(self, force: bool = False) -> None:
        try:
            mtime = os.path.getmtime(self._demo4_file)
            if not force and mtime == self._demo4_mtime:
                return
            with open(self._demo4_file, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            schema = str(data.get("schema_version", ""))
            target = str(
                data.get("mission_target", data.get("target_object", ""))
            ).strip().lower()
            if schema and schema != "1.0":
                raise ValueError(f"不支持的 Demo4 schema_version={schema}")
            if target not in ("red box", "red cube", "red block"):
                raise ValueError(f"Demo4 任务目标不是 red box: {target or '<空>'}")
            self._demo4_mtime = mtime
            self._warned_no_target = False
            self._warned_bad_demo4 = False
        except FileNotFoundError:
            if self._fallback_target:
                self._target = self._fallback_target
            elif force:
                self.get_logger().warn(f"等待 Demo4 结果文件: {self._demo4_file}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if not self._warned_bad_demo4:
                self.get_logger().warn(f"读取 Demo4 结果失败，保留上一目标: {exc}")
                self._warned_bad_demo4 = True

    def _publish_completed(self) -> None:
        if self._future is None or not self._future.done():
            return
        completed = self._future
        self._future = None
        try:
            result = completed.result()
        except Exception as exc:
            self.get_logger().error(f"Qwen 检测任务异常: {exc}")
            return

        self._publish_result(result, self._future_source_stamp)

    def _publish_result(self, result: Dict[str, Any], source_stamp: Dict[str, int]) -> None:
        result["source_stamp"] = dict(source_stamp)
        message = String()
        message.data = json.dumps(result, ensure_ascii=False)
        self._publisher.publish(message)
        observation_status = str(result.get("observation_status", ""))
        if result["success"]:
            self.get_logger().info(
                f"检测到 {result['target']}: "
                f"conf={result['confidence']:.3f}, "
                f"center=({result['center_pixel']['u']}, "
                f"{result['center_pixel']['v']}), "
                f"detector={result.get('detector', 'qwen_vl')}"
            )
        elif observation_status == "not_found":
            self.get_logger().warn(
                "当前视野未检测到 red box，等待大脑执行有界搜索",
                throttle_duration_sec=5.0,
            )
        else:
            self.get_logger().error(
                "本次检测无有效结论，自动控制应保持停车: "
                f"{result.get('error', result.get('reason', 'unknown error'))}",
                throttle_duration_sec=5.0,
            )

    def _schedule_detection(self) -> None:
        self._refresh_target()

        # A recent high-rate local confirmation is sufficient; keep cloud VLM
        # calls at the low detect_rate and do not spend quota on the same frame.
        if time.monotonic() - self._last_local_success_at < 1.0:
            return

        if self._future is not None:
            return

        if not self._target:
            if not self._warned_no_target:
                self.get_logger().warn("尚无检测目标，请先运行 Demo4")
                self._warned_no_target = True
            return

        if not self._api_key and not self._enable_color_fallback:
            if not self._warned_no_key:
                self.get_logger().warn("没有 DASHSCOPE_API_KEY，且本地兜底关闭")
                self._warned_no_key = True
            return

        with self._frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame
            frame_id = self._latest_frame_id
            source_stamp = dict(self._latest_source_stamp)
            self._latest_frame = None

        self._frame_number += 1
        self._future = self._executor.submit(
            self._detect, frame, self._target, frame_id, self._frame_number
        )
        self._future_source_stamp = source_stamp

    def _detect(
        self, image: np.ndarray, target: str, frame_id: str, frame_number: int
    ) -> Dict[str, Any]:
        height, width = image.shape[:2]

        qwen_error = ""
        if self._api_key:
            try:
                result = self._detect_with_qwen(image, target, frame_id, frame_number)
                if result is not None:
                    return result
            except Exception as exc:
                qwen_error = str(exc)
                if not self._warned_vlm_failure:
                    self.get_logger().warn(f"Qwen 检测失败，尝试本地颜色兜底: {exc}")
                    self._warned_vlm_failure = True
        else:
            qwen_error = "DASHSCOPE_API_KEY is not set"

        if self._enable_color_fallback:
            return self._negative_detection(
                frame_id,
                frame_number,
                width,
                height,
                "local_color_fallback",
                "red box is not visible in the current frame",
                observation_status="not_found",
                upstream_error=qwen_error,
            )

        return self._negative_detection(
            frame_id,
            frame_number,
            width,
            height,
            "qwen_vl",
            "detector failed before producing a visual conclusion",
            observation_status="error",
            error=qwen_error or "detector returned no valid result",
        )

    @staticmethod
    def _negative_detection(
        frame_id: str,
        frame_number: int,
        width: int,
        height: int,
        detector: str,
        reason: str,
        observation_status: str = "not_found",
        error: str = "",
        upstream_error: str = "",
    ) -> Dict[str, Any]:
        """Build the same Demo5 schema for every negative detection."""
        return {
            "schema_version": "1.0",
            "success": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image": f"ros_frame_{frame_number}",
            "target": "red box",
            "bbox_pixel": None,
            "center_pixel": None,
            "position": "unknown",
            "confidence": 0.0,
            "image_width": width,
            "image_height": height,
            "frame_id": frame_id,
            "detector": detector,
            "observation_status": observation_status,
            "reason": reason,
            "error": error,
            "upstream_error": upstream_error,
        }

    def _detect_with_qwen(
        self, image: np.ndarray, target: str, frame_id: str, frame_number: int
    ) -> Optional[Dict[str, Any]]:
        height, width = image.shape[:2]
        tmp_path = ""
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="demo5_qwen_", suffix=".jpg")
            os.close(fd)
            scale = min(
                1.0,
                self._cloud_image_max_width / max(1, width),
                self._cloud_image_max_height / max(1, height),
            )
            upload_image = image
            if scale < 1.0:
                upload_image = cv2.resize(
                    image,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            upload_height, upload_width = upload_image.shape[:2]
            if not cv2.imwrite(
                tmp_path,
                upload_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
            ):
                raise RuntimeError("保存临时推理图像失败")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": tmp_path},
                        {"text": build_bbox_prompt(target, upload_width, upload_height)},
                    ],
                }
            ]
            response = MultiModalConversation.call(
                model=self._vlm_model,
                api_key=self._api_key,
                messages=messages,
                max_tokens=max(64, min(int(self._max_tokens), 2048)),
                enable_thinking=False,
            )
            parsed = parse_json(extract_response_text(response))
            if not parsed:
                raise RuntimeError("Qwen 返回内容不是有效 JSON")
            # ``success=false`` is a valid visual observation: the semantic
            # target is simply outside the current camera view.  Preserve it
            # as ``not_found`` so the supervisor continues its bounded
            # stop-and-look search instead of reporting a detector failure.
            if not bool(parsed.get("success", True)):
                return self._negative_detection(
                    frame_id,
                    frame_number,
                    width,
                    height,
                    "qwen_vl",
                    "red box is not visible in the current Qwen frame",
                    observation_status="not_found",
                )
            if scale < 1.0 and isinstance(parsed.get("bbox_pixel"), dict):
                bbox = parsed["bbox_pixel"]
                for key in ("x1", "x2"):
                    bbox[key] = float(bbox[key]) / scale
                for key in ("y1", "y2"):
                    bbox[key] = float(bbox[key]) / scale
            normalized = self._normalize_detection(
                parsed, target, frame_id, frame_number, width, height, "qwen_vl"
            )
            if normalized is None:
                raise RuntimeError("Qwen 返回的 red-box bbox 无效或低于置信度阈值")
            return normalized
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _normalize_detection(
        self,
        parsed: Dict[str, Any],
        target: str,
        frame_id: str,
        frame_number: int,
        width: int,
        height: int,
        detector: str,
    ) -> Optional[Dict[str, Any]]:
        if not bool(parsed.get("success", True)):
            return None

        parsed_target = str(parsed.get("target", target)).strip().lower()
        if parsed_target not in ("red box", "red cube", "red block"):
            return None

        bbox = parsed.get("bbox_pixel") or parsed.get("bbox") or {}
        if not isinstance(bbox, dict):
            return None
        try:
            x1 = int(round(float(bbox["x1"])))
            y1 = int(round(float(bbox["y1"])))
            x2 = int(round(float(bbox["x2"])))
            y2 = int(round(float(bbox["y2"])))
            confidence = float(parsed.get("confidence", 0.5))
        except (KeyError, TypeError, ValueError):
            return None

        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if x2 <= x1 or y2 <= y1 or confidence < self._threshold:
            return None

        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        horizontal = "左侧" if u < width / 3 else "右侧" if u > width * 2 / 3 else "中央"
        vertical = "上方" if v < height / 3 else "下方" if v > height * 2 / 3 else "中间"
        return {
            "schema_version": "1.0",
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image": f"ros_frame_{frame_number}",
            "target": "red box",
            "bbox_pixel": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center_pixel": {"u": u, "v": v},
            "position": vertical + horizontal,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "image_width": width,
            "image_height": height,
            "frame_id": frame_id,
            "detector": detector,
            "observation_status": "detected",
            "reason": "red box detected",
            "error": "",
        }

    def _detect_colored_box(
        self, image: np.ndarray, target: str, frame_id: str, frame_number: int
    ) -> Optional[Dict[str, Any]]:
        target_l = target.lower()
        if "red" not in target_l:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        normalized_target = "red box"

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        height, width = image.shape[:2]
        min_area = max(80.0, width * height * 0.0005)
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        confidence = min(0.99, max(self._threshold, area / float(width * height) * 20.0))
        parsed = {
            "success": True,
            "target": normalized_target,
            "bbox_pixel": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
            "confidence": confidence,
        }
        return self._normalize_detection(
            parsed, normalized_target, frame_id, frame_number, width, height, "local_color_fallback"
        )

    def destroy_node(self) -> bool:
        self._detect_timer.cancel()
        self._result_timer.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = Demo5QwenDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("节点被用户中断")
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
