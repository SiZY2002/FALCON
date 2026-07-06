# 仓库贡献指南

## 项目结构与模块组织

FALCON 是一个用于力自适应人形机器人移动操作的 Python 机器人项目。核心训练与评估代码位于 `humanoidverse/`：智能体代码在 `humanoidverse/agents`，任务与仿真器代码在 `humanoidverse/envs` 和 `humanoidverse/simulator`，Hydra 配置在 `humanoidverse/config`，机器人与动作资源在 `humanoidverse/data`。部署流程位于 `sim2real/`，其中 `sim2real/config` 存放配置，`sim2real/models` 存放 ONNX 策略，`sim2real/sim_env` 提供 Mujoco 入口，`sim2real/rl_policy` 提供策略启动脚本。共享 Isaac 工具位于 `isaac_utils/`。项目图片和文档资源位于 `assets/`。

## 构建、测试与开发命令

- `pip install -e .`：以可编辑模式安装主 `falcon` Python 包。
- `pip install -e isaac_utils`：安装本地 Isaac 辅助工具。
- `cd sim2real && pip install -r requirements.txt`：安装部署相关依赖。
- `python humanoidverse/train_agent.py +exp=<exp> +simulator=isaacgym ...`：启动基于 Hydra 的训练；可参考 `README.md` 中的完整示例。
- `python humanoidverse/eval_agent.py +checkpoint=<path_to_ckpt>`：评估已训练的 checkpoint。
- `cd sim2real && python sim_env/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml`：启动 FALCON 的 sim2sim Mujoco 仿真。
- `cd sim2real && python rl_policy/loco_manip/loco_manip.py --config=config/g1/g1_29dof_falcon.yaml --model_path=models/falcon/g1_29dof.onnx`：启动部署策略。

## 代码风格与命名约定

训练代码使用 Python 3.8+，`sim2real` 使用 Python 3.10，与 README 保持一致。遵循现有风格：4 空格缩进，模块、函数和变量使用 `snake_case`，类名使用 `PascalCase`，YAML 配置文件名应体现机器人、任务和观测变体。优先使用 Hydra 配置覆盖实验参数，避免在代码中硬编码实验常量。机器人资源和生成的策略文件应保留在现有 data/model 目录中。

## 测试指南

当前仓库没有独立测试套件。提交前请做有针对性的 smoke test：导入修改过的模块，在可行时用小规模或 debug 配置运行相关训练或评估入口，并对部署改动运行对应的 `sim2real` 启动脚本。修改资源或配置时，需要同时确认路径能从仓库根目录和 `sim2real/` 目录正确解析。

## Commit 与 Pull Request 规范

近期提交使用简短的祈使式摘要，例如 `fix onnx inference in sim2real` 和 `add keyboard shortcuts example`。每个 commit 应聚焦一个行为变更或资源更新。PR 需要说明受影响的机器人或配置，列出已运行的命令，注明 IsaacGym、Mujoco 或机器人 SDK 等必需依赖；如果改动影响仿真器或策略行为，建议附上截图或视频片段。

## 安全与配置提示

不要提交私有机器人网络设置、凭据或本机专用路径。运行 sim2real 前，请检查所选 `sim2real/config/*.yaml` 中的 `ROBOT_TYPE`、`SDK_TYPE`、`INTERFACE` 和 `DOMAIN_ID`。
