import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy


class MotionRefLocoManipPolicy(LocoManipPolicy):
    def __init__(
        self,
        config,
        model_path,
        motion_path,
        motion_fps=30.0,
        rl_rate=50,
        policy_action_scale=0.25,
    ):
        config = dict(config)
        config["use_upper_body_controller"] = False
        super().__init__(config, model_path, rl_rate, policy_action_scale)

        self.motion_fps = float(motion_fps)
        if self.motion_fps <= 0.0:
            raise ValueError("motion_fps must be positive")

        self.motion_path = self._resolve_motion_path(motion_path)
        self.upper_motion = self._load_upper_motion(self.motion_path)
        self.motion_start_time = time.perf_counter()
        self.motion_frame_index = -1
        self.logger.info(
            f"Loaded upper body motion from {self.motion_path}: "
            f"{self.upper_motion.shape[0]} frames at {self.motion_fps:.2f} FPS"
        )

    def _resolve_motion_path(self, motion_path):
        if os.path.isabs(motion_path):
            return motion_path

        current_dir = os.path.dirname(__file__)
        sim2real_dir = os.path.abspath(os.path.join(current_dir, "../../"))
        repo_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
        candidates = [
            os.path.abspath(motion_path),
            os.path.abspath(os.path.join(sim2real_dir, motion_path)),
            os.path.abspath(os.path.join(repo_dir, motion_path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _load_upper_motion(self, motion_path):
        motion = np.loadtxt(motion_path, delimiter=",", dtype=np.float32)
        if motion.ndim == 1:
            motion = motion.reshape(1, -1)
        if motion.shape[1] < 36:
            raise ValueError(
                f"Motion file must have at least 36 columns, got {motion.shape[1]}: {motion_path}"
            )

        # CSV columns 22:36 are G1 arm joint angles. In 1-based counting this is columns 23:36.
        upper_motion = motion[:, 22:36]
        if upper_motion.shape[1] != self.num_upper_dofs:
            raise ValueError(
                f"Expected {self.num_upper_dofs} upper body joints, got {upper_motion.shape[1]}"
            )
        if not np.isfinite(upper_motion).all():
            raise ValueError(f"Motion file contains NaN or Inf values: {motion_path}")

        return upper_motion

    def _update_ref_upper_dof_pos_from_motion(self):
        elapsed = time.perf_counter() - self.motion_start_time
        frame_index = int(elapsed * self.motion_fps) % self.upper_motion.shape[0]
        if frame_index != self.motion_frame_index:
            self.motion_frame_index = frame_index
            self.ref_upper_dof_pos[0, :] = self.upper_motion[frame_index]

    def _handle_start_policy(self):
        self.motion_start_time = time.perf_counter()
        self.motion_frame_index = -1
        self._update_ref_upper_dof_pos_from_motion()
        super()._handle_start_policy()

    def policy_action(self):
        self._update_ref_upper_dof_pos_from_motion()
        super().policy_action()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    parser.add_argument(
        "--motion_path",
        type=str,
        default="upper_data/robot_motion.csv",
        help="path to the CSV motion file",
    )
    parser.add_argument("--motion_fps", type=float, default=30.0, help="motion playback FPS")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = MotionRefLocoManipPolicy(
        config=config,
        model_path=model_path,
        motion_path=args.motion_path,
        motion_fps=args.motion_fps,
        rl_rate=50,
        policy_action_scale=0.25,
    )
    policy.run()
