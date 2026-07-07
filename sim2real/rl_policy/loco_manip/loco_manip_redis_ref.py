import argparse
import json
import socket
import sys
import time

import numpy as np
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy


G1_UPPER_DOF_START = 15
G1_UPPER_DOF_END = 29
G1_UPPER_DOF_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class LocalRedisClient:
    def __init__(self, host="localhost", port=6379, timeout_s=0.02):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock = None
        self.reader = None

    def close(self):
        if self.reader is not None:
            self.reader.close()
        if self.sock is not None:
            self.sock.close()
        self.reader = None
        self.sock = None

    def get(self, key):
        try:
            return self._command("GET", key)
        except (OSError, ValueError):
            self.close()
            return None

    def _connect(self):
        if self.sock is not None:
            return
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self.sock.settimeout(self.timeout_s)
        self.reader = self.sock.makefile("rb")

    def _command(self, *parts):
        self._connect()
        payload = self._encode_command(parts)
        self.sock.sendall(payload)
        return self._read_response()

    @staticmethod
    def _encode_command(parts):
        encoded = [f"*{len(parts)}\r\n".encode("ascii")]
        for part in parts:
            if isinstance(part, bytes):
                value = part
            else:
                value = str(part).encode("utf-8")
            encoded.append(f"${len(value)}\r\n".encode("ascii"))
            encoded.append(value + b"\r\n")
        return b"".join(encoded)

    def _read_line(self):
        line = self.reader.readline()
        if not line:
            raise ValueError("empty Redis response")
        return line.rstrip(b"\r\n")

    def _read_response(self):
        prefix = self.reader.read(1)
        if not prefix:
            raise ValueError("empty Redis response")
        if prefix == b"+":
            return self._read_line()
        if prefix == b"-":
            raise ValueError(self._read_line().decode("utf-8", errors="replace"))
        if prefix == b":":
            return int(self._read_line())
        if prefix == b"$":
            length = int(self._read_line())
            if length == -1:
                return None
            data = self.reader.read(length)
            crlf = self.reader.read(2)
            if len(data) != length or crlf != b"\r\n":
                raise ValueError("invalid Redis bulk string")
            return data
        raise ValueError(f"unsupported Redis response prefix: {prefix!r}")


class RedisRefLocoManipPolicy(LocoManipPolicy):
    def __init__(
        self,
        config,
        model_path,
        redis_ip="localhost",
        redis_port=6379,
        redis_key="retarget_joint_angles_unitree_g1",
        max_data_age_s=0.2,
        redis_timeout_s=0.02,
        warn_interval_s=1.0,
        rl_rate=50,
        policy_action_scale=0.25,
    ):
        config = dict(config)
        config["use_upper_body_controller"] = False
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.redis_key = redis_key
        self.redis_time_key = "t_" + redis_key
        self.max_data_age_s = max_data_age_s
        self.warn_interval_s = warn_interval_s
        self.last_warn_time = 0.0
        self.last_status_time = 0.0
        self.last_data_age_s = None
        self.redis_update_count = 0
        self.redis_client = LocalRedisClient(redis_ip, redis_port, redis_timeout_s)

        self._validate_upper_dof_order()
        self.logger.info(
            f"Reading G1 joint angles from Redis {redis_ip}:{redis_port}, key={redis_key}"
        )

    def _validate_upper_dof_order(self):
        if self.upper_dof_names != G1_UPPER_DOF_NAMES:
            raise ValueError(
                "Upper DOF order mismatch. Expected "
                f"{G1_UPPER_DOF_NAMES}, got {self.upper_dof_names}"
            )

    def _warn_throttled(self, message):
        now = time.perf_counter()
        if now - self.last_warn_time >= self.warn_interval_s:
            self.last_warn_time = now
            self.logger.warning(message)

    def _status_throttled(self, upper_joint_angles):
        now = time.perf_counter()
        if now - self.last_status_time < self.warn_interval_s:
            return

        self.last_status_time = now
        age_text = "unknown" if self.last_data_age_s is None else f"{self.last_data_age_s:.3f}s"
        self.logger.info(
            f"Redis ref_upper_dof_pos updated: count={self.redis_update_count}, "
            f"age={age_text}, min={upper_joint_angles.min():.4f}, max={upper_joint_angles.max():.4f}"
        )

    def _read_joint_angles(self):
        raw = self.redis_client.get(self.redis_key)
        if raw is None:
            self._warn_throttled(f"No Redis data on key: {self.redis_key}")
            return None

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._warn_throttled(f"Invalid Redis JSON on {self.redis_key}: {exc}")
            return None

        if isinstance(payload, dict):
            try:
                joint_angles = np.array([payload[name] for name in G1_UPPER_DOF_NAMES], dtype=np.float32)
                return joint_angles
            except KeyError as exc:
                self._warn_throttled(f"Missing upper joint in Redis dict payload: {exc}")
                return None

        joint_angles = np.asarray(payload, dtype=np.float32)
        if joint_angles.shape != (29,):
            self._warn_throttled(
                f"Expected 29 G1 joint angles from Redis, got shape {joint_angles.shape}"
            )
            return None
        if not np.isfinite(joint_angles).all():
            self._warn_throttled("Redis joint angle payload contains NaN or Inf")
            return None

        return joint_angles[G1_UPPER_DOF_START:G1_UPPER_DOF_END]

    def _is_data_fresh(self):
        if self.max_data_age_s <= 0.0:
            return True

        raw_timestamp = self.redis_client.get(self.redis_time_key)
        if raw_timestamp is None:
            self._warn_throttled(f"No Redis timestamp on key: {self.redis_time_key}")
            return False

        try:
            timestamp_ms = int(raw_timestamp)
        except ValueError:
            self._warn_throttled(f"Invalid Redis timestamp on key: {self.redis_time_key}")
            return False

        age_s = time.time() - timestamp_ms / 1000.0
        self.last_data_age_s = age_s
        if age_s > self.max_data_age_s:
            self._warn_throttled(
                f"Redis joint angle data is stale: age={age_s:.3f}s, max={self.max_data_age_s:.3f}s"
            )
            return False
        return True

    def _update_ref_upper_dof_pos_from_redis(self):
        if not self._is_data_fresh():
            return

        upper_joint_angles = self._read_joint_angles()
        if upper_joint_angles is None:
            return

        self.ref_upper_dof_pos[0, :] = upper_joint_angles
        self.redis_update_count += 1
        self._status_throttled(upper_joint_angles)

    def policy_action(self):
        self._update_ref_upper_dof_pos_from_redis()
        super().policy_action()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--redis_key",
        type=str,
        default="retarget_joint_angles_unitree_g1",
        help="Redis key containing 29 absolute G1 joint angles in rad",
    )
    parser.add_argument(
        "--max_data_age_s",
        type=float,
        default=0.2,
        help="Maximum accepted age for Redis data. Use <=0 to disable stale checks.",
    )
    parser.add_argument("--redis_timeout_s", type=float, default=0.02, help="Redis socket timeout")
    parser.add_argument("--warn_interval_s", type=float, default=1.0, help="Warning throttle interval")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = RedisRefLocoManipPolicy(
        config=config,
        model_path=model_path,
        redis_ip=args.redis_ip,
        redis_port=args.redis_port,
        redis_key=args.redis_key,
        max_data_age_s=args.max_data_age_s,
        redis_timeout_s=args.redis_timeout_s,
        warn_interval_s=args.warn_interval_s,
        rl_rate=50,
        policy_action_scale=0.25,
    )
    policy.run()
