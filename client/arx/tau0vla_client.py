"""Independent ARX1 ROS client for remote Tau0VLA action-chunk inference."""

from __future__ import annotations

import argparse
import collections
import json
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from safe_height import is_safe_and_stable
from tau0vla_protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    FPS,
    ActionEMA,
    ActionChunk,
    ChunkScheduler,
    Observation,
    ProtocolError,
    RollingDelayEstimator,
    Tau0VLAHttpClient,
    recommended_replan_steps,
)
from tau0vla_trace import TraceWriter, analyze_trace
from utils.setup_loader import setup_loader


FILE = Path(__file__).resolve()
ROOT = FILE.parent
DEFAULT_TASK = "Pick up the handle and place it into the tray."


def create_observation_node(config: dict, *, max_observation_age_ms: float, max_camera_skew_ms: float):
    """Create a small ROS node that preserves compressed images and joint feedback."""
    from rclpy.node import Node

    class ObservationNode(Node):
        def __init__(self):
            super().__init__("tau0vla_arx_client")
            from arm_control.msg import PosCmd
            from arx5_arm_msg.msg import RobotStatus
            from sensor_msgs.msg import CompressedImage

            self._lock = threading.Lock()
            self._latest = {}
            self._height_samples = collections.deque(maxlen=2000)
            self._robot_status_type = RobotStatus
            self._left_publisher = None
            self._right_publisher = None
            self._max_age_ns = int(max_observation_age_ms * 1_000_000)
            self._max_camera_skew_ns = int(max_camera_skew_ms * 1_000_000)
            cameras = {
                "head": config["camera_config"]["img_head_topic"],
                "left_wrist": config["camera_config"]["img_left_topic"],
                "right_wrist": config["camera_config"]["img_right_topic"],
            }
            for name, topic in cameras.items():
                self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda message, key=name: self._receive(f"camera:{key}", message),
                    10,
                )
            self.create_subscription(
                RobotStatus,
                config["arm_config"]["follow_arm_left_feedback_topic"],
                lambda message: self._receive("arm:left", message),
                10,
            )
            self.create_subscription(
                RobotStatus,
                config["arm_config"]["follow_arm_right_feedback_topic"],
                lambda message: self._receive("arm:right", message),
                10,
            )
            self.create_subscription(
                PosCmd,
                config["robot_base_config"]["robot_base_topic"],
                self._receive_height,
                10,
            )

        def _receive(self, key: str, message) -> None:
            with self._lock:
                self._latest[key] = (message, time.monotonic_ns())

        def _receive_height(self, message) -> None:
            with self._lock:
                self._height_samples.append((time.monotonic(), float(message.height)))

        @staticmethod
        def _stamp_ns(message) -> int:
            stamp = message.header.stamp
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        def snapshot(self) -> Observation:
            required = [
                "camera:head",
                "camera:left_wrist",
                "camera:right_wrist",
                "arm:left",
                "arm:right",
            ]
            now_ns = time.monotonic_ns()
            with self._lock:
                missing = [key for key in required if key not in self._latest]
                if missing:
                    raise ProtocolError(f"missing ROS observations: {missing}")
                latest = {key: self._latest[key] for key in required}
            stale = [key for key, (_, received_ns) in latest.items() if now_ns - received_ns > self._max_age_ns]
            if stale:
                raise ProtocolError(f"stale ROS observations: {stale}")
            camera_stamps = [
                self._stamp_ns(latest[f"camera:{name}"][0])
                for name in ("head", "left_wrist", "right_wrist")
            ]
            if max(camera_stamps) - min(camera_stamps) > self._max_camera_skew_ns:
                raise ProtocolError("camera timestamp skew exceeds configured limit")
            left = np.asarray(latest["arm:left"][0].joint_pos, dtype=np.float32)
            right = np.asarray(latest["arm:right"][0].joint_pos, dtype=np.float32)
            qpos = np.concatenate((left[:7], right[:7]))
            if qpos.shape != (ACTION_DIM,) or not np.isfinite(qpos).all():
                raise ProtocolError("ROS joint feedback is not a finite 14-vector")
            images = {
                name: bytes(latest[f"camera:{name}"][0].data)
                for name in ("head", "left_wrist", "right_wrist")
            }
            if any(not value for value in images.values()):
                raise ProtocolError("ROS camera message contains an empty JPEG")
            return Observation(qpos=qpos, images=images, sample_monotonic_ns=now_ns)

        def height_samples(self):
            with self._lock:
                return tuple(self._height_samples)

        def current_qpos(self) -> np.ndarray:
            with self._lock:
                if "arm:left" not in self._latest or "arm:right" not in self._latest:
                    raise ProtocolError("joint feedback is unavailable")
                left = np.asarray(self._latest["arm:left"][0].joint_pos, dtype=np.float32)
                right = np.asarray(self._latest["arm:right"][0].joint_pos, dtype=np.float32)
            qpos = np.concatenate((left[:7], right[:7]))
            if qpos.shape != (ACTION_DIM,) or not np.isfinite(qpos).all():
                raise ProtocolError("joint feedback is not a finite 14-vector")
            return qpos

        def enable_publishers(self) -> None:
            if self._left_publisher is not None:
                return
            self._left_publisher = self.create_publisher(
                self._robot_status_type,
                config["arm_config"]["follow_arm_left_cmd_topic"],
                10,
            )
            self._right_publisher = self.create_publisher(
                self._robot_status_type,
                config["arm_config"]["follow_arm_right_cmd_topic"],
                10,
            )

        def publish_action(self, action: np.ndarray) -> None:
            if self._left_publisher is None or self._right_publisher is None:
                raise RuntimeError("action publishers are disabled")
            values = np.asarray(action, dtype=np.float32)
            if values.shape != (ACTION_DIM,) or not np.isfinite(values).all():
                raise ProtocolError("refusing malformed action at ROS publication boundary")
            left = self._robot_status_type()
            right = self._robot_status_type()
            left.joint_pos[:7] = [float(value) for value in values[:7]]
            right.joint_pos[:7] = [float(value) for value in values[7:]]
            self._left_publisher.publish(left)
            self._right_publisher.publish(right)

    return ObservationNode()


def spin_node(node, stop_event: threading.Event) -> None:
    import rclpy

    while rclpy.ok() and not stop_event.is_set():
        rclpy.spin_once(node, timeout_sec=0.01)


def verify_height(node, expected_height: float, tolerance: float, window: float, timeout: float) -> float:
    from rclpy.parameter_client import AsyncParameterClient

    client = AsyncParameterClient(node, "/lift")
    if not client.wait_for_services(timeout_sec=5.0):
        raise RuntimeError("/lift parameter service is unavailable")
    future = client.get_parameters(["fixed_height"])
    deadline = time.monotonic() + 5.0
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not future.done() or future.result() is None:
        raise RuntimeError("timed out reading /lift fixed_height")
    fixed_height = float(future.result().values[0].double_value)
    if not np.isclose(fixed_height, expected_height, atol=1e-6):
        raise RuntimeError(f"/lift fixed_height={fixed_height}, expected {expected_height}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples = node.height_samples()
        if is_safe_and_stable(samples, float("inf"), tolerance, window):
            feedback = float(samples[-1][1])
            print(f"Height verified: fixed_height={fixed_height:.6f}, stable_feedback={feedback:.6f}")
            return feedback
        time.sleep(0.05)
    raise RuntimeError("height feedback did not become stable")


def wait_for_observation(node, timeout: float) -> Observation:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return node.snapshot()
        except ProtocolError as error:
            last_error = error
            time.sleep(0.05)
    raise RuntimeError(f"sensor preflight timed out: {last_error}")


def benchmark(client: Tau0VLAHttpClient, node, request_id: int, warmup: int, samples: int):
    latencies = []
    total = warmup + samples
    for index in range(total):
        request_id += 1
        result = client.infer(wait_for_observation(node, 5.0), request_id)
        if index >= warmup:
            latencies.append(result.round_trip_ms)
        print(
            f"Benchmark {index + 1}/{total}: request={request_id}, "
            f"RTT={result.round_trip_ms:.1f} ms, inference={result.inference_ms:.1f} ms"
        )
    return request_id, latencies


def _resolve_replan_steps(value: str, latencies: list[float], margin_ms: float) -> tuple[int, float]:
    automatic, p99_ms = recommended_replan_steps(latencies, margin_ms=margin_ms)
    if value == "auto":
        return automatic, p99_ms
    selected = int(value)
    if not 1 <= selected < ACTION_HORIZON:
        raise ValueError(f"--replan-steps must be auto or an integer in [1, {ACTION_HORIZON - 1}]")
    available_ms = (ACTION_HORIZON - selected) * 1000.0 / FPS
    if p99_ms + margin_ms >= available_ms:
        raise ProtocolError(
            f"replan_steps={selected} leaves {available_ms:.1f} ms, below "
            f"p99 RTT + margin={p99_ms + margin_ms:.1f} ms"
        )
    return selected, p99_ms


def run(args) -> None:
    import rclpy

    setup_loader(ROOT)
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    rclpy.init()
    node = create_observation_node(
        config,
        max_observation_age_ms=args.max_observation_age_ms,
        max_camera_skew_ms=args.max_camera_skew_ms,
    )
    stop_event = threading.Event()
    spin_thread = threading.Thread(target=spin_node, args=(node, stop_event), daemon=True)
    spin_thread.start()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    executor = ThreadPoolExecutor(max_workers=1)
    pending: Future[ActionChunk] | None = None
    trace = TraceWriter(None)
    try:
        trace = TraceWriter(args.trace_path)
        verify_height(
            node,
            args.expected_height,
            args.height_stability_tolerance,
            args.height_stability_window,
            args.height_timeout,
        )
        wait_for_observation(node, args.sensor_timeout)
        client = Tau0VLAHttpClient(
            args.server_url,
            request_timeout=args.request_timeout,
            max_response_age_ms=args.max_response_age_ms,
            rtc=True,
        )
        health = client.health()
        contract = client.policy_contract()
        session = client.create_session(args.task_instruction)
        print(f"Server health: {json.dumps(health, sort_keys=True)}")
        print(f"Policy contract: {json.dumps(contract, sort_keys=True)}")
        print(f"Session: {session['session_id']}")
        trace.metadata(
            session_id=session["session_id"],
            server_url=args.server_url,
            task_instruction=args.task_instruction,
            execute=args.execute,
            blend_steps=args.chunk_blend_steps,
            arm_ema_alpha=args.arm_ema_alpha,
            gripper_ema_alpha=args.gripper_ema_alpha,
            rtc_enabled=True,
            rtc_max_delay=client.rtc_max_delay,
        )

        request_id, latencies = benchmark(
            client,
            node,
            request_id=0,
            warmup=args.benchmark_warmup,
            samples=args.benchmark_requests,
        )
        replan_steps, p99_ms = _resolve_replan_steps(args.replan_steps, latencies, args.latency_margin_ms)
        print(f"Selected replan_steps={replan_steps}; measured p99 RTT={p99_ms:.1f} ms")
        print(f"Runtime response age limit={client.max_response_age_ms:.1f} ms")
        delay_estimator = RollingDelayEstimator(latencies, margin_ms=args.latency_margin_ms, window_size=30)
        initial_delay = delay_estimator.predict()
        print(
            f"RTC rolling delay={initial_delay} steps; checkpoint maximum={client.rtc_max_delay} steps"
        )

        request_id += 1
        first = client.infer(wait_for_observation(node, 5.0), request_id)
        scheduler = ChunkScheduler(replan_steps, blend_steps=args.chunk_blend_steps)
        ema = ActionEMA(args.arm_ema_alpha, args.gripper_ema_alpha)
        ema.reset(node.current_qpos())
        first_info = scheduler.adopt(first, initial=True, arrival_monotonic_ns=time.monotonic_ns())
        trace.adoption(first.request_id, first_info)

        if args.execute:
            confirmation = input(
                "Workspace clear and emergency stop reachable. "
                "Type EXECUTE TAU0VLA PICKPLACE to publish model actions: "
            )
            if confirmation != "EXECUTE TAU0VLA PICKPLACE":
                raise RuntimeError("execution cancelled; no action publisher was created")
            node.enable_publishers()
            print(
                "EXECUTE mode: publishing finite 14D Tau0VLA actions "
                f"with blend_steps={args.chunk_blend_steps}, arm_ema_alpha={args.arm_ema_alpha}, "
                f"gripper_ema_alpha={args.gripper_ema_alpha}."
            )
        else:
            print("DRY-RUN mode: no action publisher was created.")

        period = 1.0 / FPS
        deadline = time.monotonic()
        starvation_logged = False
        pending_control_step = 0
        step = 0
        published_since_log = 0
        publish_window_started = time.monotonic()
        while rclpy.ok() and not stop_event.is_set() and step < args.max_steps:
            now = time.monotonic()
            if now < deadline:
                time.sleep(min(deadline - now, 0.005))
                continue
            deadline += period
            if now - deadline > period:
                deadline = now + period

            if pending is not None and pending.done():
                result = pending.result()
                actual_elapsed = max(0, step - pending_control_step)
                delay_estimator.observe(max(result.round_trip_ms, actual_elapsed * 1000.0 / FPS))
                adoption = scheduler.adopt(result, actual_elapsed=actual_elapsed)
                trace.adoption(result.request_id, adoption)
                if adoption.adopted:
                    print(
                        f"RTC adopted request={result.request_id}, predicted={result.rtc_delay}, "
                        f"elapsed={actual_elapsed}, RTT={result.round_trip_ms:.1f} ms, "
                        f"prefix_error={adoption.prefix_max_abs_error:.3g}, "
                        f"blended={adoption.blended_steps}, buffer={scheduler.remaining}"
                    )
                    starvation_logged = False
                else:
                    print(
                        f"RTC DROP request={result.request_id}: {adoption.drop_reason}; "
                        f"predicted={result.rtc_delay}, elapsed={actual_elapsed}, "
                        f"next_prediction={delay_estimator.predict()}"
                    )
                pending = None

            predicted_delay = delay_estimator.predict()
            delay_is_safe = predicted_delay <= client.rtc_max_delay
            if delay_is_safe and scheduler.should_request(pending is not None, predicted_delay):
                request_id += 1
                observation = node.snapshot()
                action_prefix = scheduler.preview(predicted_delay, ema)
                pending_control_step = step
                pending = executor.submit(
                    client.infer,
                    observation,
                    request_id,
                    rtc_delay=predicted_delay,
                    action_prefix=action_prefix,
                    request_control_step=step,
                )
                trace.request(
                    request_id=request_id,
                    control_step=step,
                    predicted_delay=predicted_delay,
                    source_index=scheduler.current_source_index,
                    action_prefix=action_prefix,
                )

            try:
                scheduled = scheduler.next_action()
            except BufferError:
                if not starvation_logged:
                    print("BUFFER STARVED: publication paused until a fresh action chunk arrives.")
                    trace.starvation(time.monotonic_ns(), step)
                    starvation_logged = True
                continue
            action = ema.apply_scheduled(scheduled)
            trace.tick(
                monotonic_ns=time.monotonic_ns(),
                control_step=step,
                scheduled=scheduled,
                command=action,
                feedback=node.current_qpos(),
                execute=args.execute,
            )
            if args.execute:
                node.publish_action(action)
                published_since_log += 1
                publish_elapsed = now - publish_window_started
                if publish_elapsed >= 1.0:
                    print(
                        f"Publish rate={published_since_log / publish_elapsed:.2f} Hz, "
                        f"step={step}, buffer={scheduler.remaining}"
                    )
                    published_since_log = 0
                    publish_window_started = now
            elif step % FPS == 0:
                print(f"DRY-RUN step={step}, action={np.array2string(action, precision=4)}")
            step += 1
    finally:
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        trace.close()
        stop_event.set()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        if args.trace_path is not None and args.trace_path.exists():
            try:
                print(f"Trace: {args.trace_path}")
                print(f"Trace summary: {json.dumps(analyze_trace(args.trace_path), sort_keys=True)}")
            except Exception as error:  # trace analysis must never block ROS cleanup
                print(f"Trace analysis failed: {error}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://192.168.50.2:8000")
    parser.add_argument("--task-instruction", default=DEFAULT_TASK)
    parser.add_argument("--config", type=Path, default=ROOT / "data/config.yaml")
    parser.add_argument("--expected-height", type=float, default=15.5)
    parser.add_argument("--height-stability-tolerance", type=float, default=0.05)
    parser.add_argument("--height-stability-window", type=float, default=2.0)
    parser.add_argument("--height-timeout", type=float, default=15.0)
    parser.add_argument("--sensor-timeout", type=float, default=30.0)
    parser.add_argument("--max-observation-age-ms", type=float, default=100.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=50.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--max-response-age-ms", type=float, default=5000.0)
    parser.add_argument("--latency-margin-ms", type=float, default=100.0)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-requests", type=int, default=30)
    parser.add_argument("--replan-steps", default="auto")
    parser.add_argument("--chunk-blend-steps", type=int, default=0)
    parser.add_argument("--arm-ema-alpha", type=float, default=1.0)
    parser.add_argument("--gripper-ema-alpha", type=float, default=1.0)
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.benchmark_warmup < 0 or args.benchmark_requests < 1:
        parser.error("benchmark counts must be non-negative, with at least one measured request")
    if args.request_timeout <= 0 or args.max_response_age_ms <= 0:
        parser.error("HTTP timeouts must be positive")
    if not 0 <= args.chunk_blend_steps < ACTION_HORIZON:
        parser.error(f"--chunk-blend-steps must be in [0, {ACTION_HORIZON - 1}]")
    for name in ("arm_ema_alpha", "gripper_ema_alpha"):
        if not 0.0 < getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
