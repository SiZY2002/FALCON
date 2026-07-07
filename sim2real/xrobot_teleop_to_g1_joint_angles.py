"""
Read PICO/XRobot body tracking, retarget it to Unitree G1, and publish only the
absolute 29-DoF G1 joint angles in radians.

This intentionally stops before TWIST2 mimic-observation construction:
    XRobotStreamer -> GMR.retarget(...) -> qpos[7:36]

The published values are absolute joint positions from the retargeted G1 qpos.
They are not offset by default_angles and are not scaled policy actions.
"""
import argparse
import json
import time

import numpy as np
import redis
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import XRobotStreamer
from loop_rate_limiters import RateLimiter
from rich import print


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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


class XRobotTeleopToG1JointAngles:
    def __init__(self, args):
        self.args = args
        self.rate = RateLimiter(frequency=args.target_fps, warn=False)
        self.streamer = XRobotStreamer()
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=args.actual_human_height,
        )
        self.redis_client = None
        self.redis_pipeline = None
        self.last_status_time = time.perf_counter()
        self.no_body_frame_count = 0
        self.valid_frame_count = 0
        self.publish_count = 0

        if not args.no_redis:
            try:
                self.redis_client = redis.Redis(host=args.redis_ip, port=6379, db=0, protocol=2)
            except TypeError:
                self.redis_client = redis.Redis(host=args.redis_ip, port=6379, db=0)
            self.redis_client.ping()
            self.redis_pipeline = self.redis_client.pipeline()
            print(f"Redis connected: {args.redis_ip}:6379")

        print("Streaming absolute G1 joint angles in rad.")
        print(f"Output key: {args.redis_key}")

    def retarget_current_frame(self):
        smplx_data, _, _, controller_data, _ = self.streamer.get_current_frame()

        if controller_data is not None:
            left_key = controller_data.get("LeftController", {}).get("key_one", False)
            if left_key:
                return None, True

        if smplx_data is None:
            return None, False

        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        joint_angles_rad = np.asarray(qpos[7:36], dtype=np.float32).copy()
        if joint_angles_rad.shape[0] != len(G1_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(G1_JOINT_NAMES)} G1 joints, got {joint_angles_rad.shape[0]}"
            )

        return joint_angles_rad, False

    def publish(self, joint_angles_rad):
        if self.redis_pipeline is None:
            return False

        timestamp_ms = int(time.time() * 1000)
        payload = joint_angles_rad.tolist()
        named_payload = {
            name: float(value)
            for name, value in zip(G1_JOINT_NAMES, payload)
        }

        self.redis_pipeline.set(self.args.redis_key, json.dumps(payload))
        self.redis_pipeline.set(self.args.redis_key + "_named", json.dumps(named_payload))
        self.redis_pipeline.set("t_" + self.args.redis_key, timestamp_ms)
        self.redis_pipeline.execute()
        return True

    def print_status_if_needed(self):
        if self.args.status_interval_s <= 0:
            return

        now = time.perf_counter()
        if now - self.last_status_time < self.args.status_interval_s:
            return

        self.last_status_time = now
        print(
            "xrobot status: "
            f"valid_frames={self.valid_frame_count}, "
            f"no_body_frames={self.no_body_frame_count}, "
            f"redis_publishes={self.publish_count}"
        )

    def run(self):
        frame_count = 0
        try:
            while True:
                joint_angles_rad, should_exit = self.retarget_current_frame()
                if should_exit:
                    print("Exit requested by left controller key_one.")
                    break

                if joint_angles_rad is not None:
                    self.valid_frame_count += 1
                    if self.publish(joint_angles_rad):
                        self.publish_count += 1
                    frame_count += 1

                    if self.args.print_interval > 0 and frame_count % self.args.print_interval == 0:
                        np.set_printoptions(precision=4, suppress=True)
                        print(f"[{frame_count}] joint_angles_rad = {joint_angles_rad}")
                else:
                    self.no_body_frame_count += 1

                self.print_status_if_needed()
                self.rate.sleep()
        except KeyboardInterrupt:
            print("Interrupted by user.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis host for publishing retargeted joint angles.",
    )
    parser.add_argument(
        "--redis_key",
        type=str,
        default="retarget_joint_angles_unitree_g1",
        help="Redis key for the 29 absolute G1 joint angles in rad.",
    )
    parser.add_argument(
        "--no_redis",
        action="store_true",
        help="Do not publish to Redis; useful with --print_interval.",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.6,
        help="Actual human height used by GMR scaling.",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=100,
        help="Retargeting loop frequency.",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=100,
        help="Print every N valid frames. Use 0 to disable printing.",
    )
    parser.add_argument(
        "--status_interval_s",
        type=float,
        default=2.0,
        help="Print stream/publish status every N seconds. Use 0 to disable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    XRobotTeleopToG1JointAngles(parse_arguments()).run()
