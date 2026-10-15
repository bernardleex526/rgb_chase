# dog_follow — 机器狗 RGB 人体跟随（与平台无关）

ROS2 (Humble/Jazzy) · Jetson Orin · 第一阶段单目，第二阶段支持深度相机（ZED / D435i）

```
相机 ──image/camera_info──> follow_perception ──/follow/target──> follow_controller ──/follow/cmd_vel──> 适配节点 ──> 机器狗
                              检测(YOLOX) → ByteTrack → ReID 锁定 → 测距        PD + 限速/限加速度          m20_adapter / twist_adapter
```

换一款狗时，只需要写或配置一个**适配节点**；换相机时，只需要改 launch 文件里的话题重映射，或者改 `range` 参数。

## 目录
| 路径 | 内容 |
|---|---|
| `dog_follow/core/` | 纯算法部分，不依赖 ROS，Windows/PC 上可直接用 pytest 测试：跟踪器、ReID 特征库、目标锁定状态机、单目/深度测距、控制器 |
| `dog_follow/backends/` | YOLOX 检测、OSNet ReID、RTMPose（可选），推理用 ONNXRuntime 或 TensorRT `.engine` |
| `dog_follow/nodes/` | ROS2 节点：感知、控制、机器狗适配 |
| `dog_follow/tools/offline_demo.py` | 不需要 ROS，直接读视频或摄像头，鼠标点击选人，用来调参 |
| `config/follow.yaml` / `robots.yaml` | 参数文件 |

## 1. 准备模型（在 Orin 上执行）
所有模型都是 Apache/MIT 许可证，没有 AGPL。

```bash
mkdir -p /opt/dog_follow/models && cd /opt/dog_follow/models
# 检测：YOLOX-s 官方 ONNX（输出原始 head，节点内部自己解码）
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
/usr/src/tensorrt/bin/trtexec --onnx=yolox_s.onnx --saveEngine=yolox_s.engine --fp16

# ReID：torchreid OSNet-x0.25（MSMT17 权重），导出成输入 (1,3,256,128)、输出 (1,512) 的 ONNX
pip install torchreid   # 或 git clone KaiyangZhou/deep-person-reid
python - <<'EOF'
import torch, torchreid
m = torchreid.models.build_model('osnet_x0_25', num_classes=1, pretrained=True).eval()
torch.onnx.export(m, torch.randn(1, 3, 256, 128), 'osnet_x0_25.onnx',
                  input_names=['input'], output_names=['output'], opset_version=13)
EOF
/usr/src/tensorrt/bin/trtexec --onnx=osnet_x0_25.onnx --saveEngine=osnet_x0_25.engine --fp16
```
- TensorRT 后端需要 `tensorrt` 和 `pycuda`（JetPack 6 自带 tensorrt；pycuda 用 `pip install pycuda` 安装）。路径以 `.onnx` 结尾的模型走 ONNXRuntime。
- 注意：torchreid 的预训练权重是在 Market1501/MSMT17 数据集上训练的，商用前请确认数据集的许可条款。

## 2. 编译
```bash
cd ~/dog_follow_ws && colcon build --symlink-install && source install/setup.bash
```

## 3. 相机参数（务必实测）
- `camera_info` 必须给出标定后的内参，可以用 `ros2 run camera_calibration cameracalibrator` 标定。
- 在 `config/follow.yaml` 中设置 `camera_height`（镜头中心离地高度）和 `camera_pitch_deg`（相机俯仰角）。单目测距主要依赖脚底触地点与地平面的几何关系，这两个值不准，距离就会有系统误差。
- 测距的融合方式：脚底触地点按地平面投影、检测框高度（被跟随者的身高会在线标定），可选再加躯干关键点；之后用卡尔曼滤波平滑。人离得很近、头和脚都超出画面时，单目没有可用的几何线索，这时控制器只转向、不前进。这种情况建议配置 `pose_model`，或者等第二阶段改用深度相机。

## 4. 运行
```bash
# 先空跑（不接机器狗），观察 /follow/debug_image 和 /follow/cmd_vel
ros2 launch dog_follow follow.launch.py robot:=none image:=/camera/color/image_raw camera_info:=/camera/color/camera_info

# 选人：按画面像素坐标（可以在 rqt_image_view 中看坐标），或者按 track id
ros2 topic pub --once /follow_perception/select_pixel geometry_msgs/msg/Point "{x: 320, y: 240}"
ros2 topic pub --once /follow_perception/select_id std_msgs/msg/Int32 "{data: 3}"
ros2 service call /follow_perception/release std_srvs/srv/Trigger

# 控制器默认不输出速度，需要显式启用
ros2 service call /follow_controller/enable std_srvs/srv/SetBool "{data: true}"
```
目标状态机：`idle →（选人）→ tracking ⇄ lost →（超过 lost_timeout）→ idle`。进入 lost 后，先朝最后看到目标的方向转约 1s，2s 后原地慢速旋转搜索，8s 后停止。重新找回目标时，候选人必须连续 3 帧与目标的外观相似度达到 `reid_accept`；如果原来的 track ID 重新出现，也要过同样的外观校验。

## 5. 云深处 M20 / M20 Pro
采用官方 [sdk_deploy](https://github.com/DeepRoboticsLab/sdk_deploy) 的 ROS2 指令模式。
1. 狗端：把 `main.cpp` 中的 `RemoteCommandType` 改成 `kRos2`，按官方 README 编译 `m20_sdk_deploy`，然后运行 `rl_deploy`。进入 SDK 模式需要手柄固件 1.1.8 以上，并向厂商技术支持申请授权码。**厂商声明：因使用 SDK 造成的损坏不在保修范围内。**
2. Orin 端：从 sdk_deploy 复制 `src/drdds` 到本工作空间一起编译（适配节点要读 `/rl_deploy/status`），并把 `ROS_DOMAIN_ID` 设成和狗端一致。狗端运行的是 ROS2 Foxy，Orin 端是 Humble，std/geometry 消息可以互通，第一次联调时请用 `ros2 topic echo /rl_deploy/status` 确认能收到。
3. 启动：
```bash
ros2 launch dog_follow follow.launch.py robot:=m20
ros2 service call /m20_adapter/stand std_srvs/srv/Trigger       # 站立
ros2 service call /m20_adapter/rl_control std_srvs/srv/Trigger  # 站稳后切到策略控制，此后速度指令才生效
ros2 service call /follow_controller/enable std_srvs/srv/SetBool "{data: true}"
# 急停（锁定输出 0 速度）：ros2 service call /m20_adapter/estop std_srvs/srv/Trigger
```
- 适配节点以 50Hz 向 `/rl_deploy/cmd_vel` 发送速度（best_effort）；狗端超过 0.25s 没收到指令会自动归零速度。
- 狗端 SDK 自带限速：前进 0.7 m/s、横向 0.5 m/s、转向 0.7 rad/s。本包默认设得更保守：0.5 m/s、0.6 rad/s。
- 厂商 README 说明实机只验证过站立和趴下，并提到"偏航超调、零输入漂移"的问题。首次上机请把狗**吊起来**或者在空旷场地测试，并把 `max_vx` 降到 0.2。

## 6. 适配其他机器狗
- 狗本身接受 `geometry_msgs/Twist`（例如 Unitree go2_ros2_sdk、Nav2 底盘）：使用 `robot:=twist`，在 `robots.yaml` 中修改 `output_topic`。
- 只有私有 SDK（UDP 或 C++ 接口）：仿照 `M20Adapter` 继承 `TwistAdapter`，重写发布部分。限幅、超时和急停逻辑都可以直接复用。

## 7. 第二阶段：深度相机
```bash
# RealSense D435i（需要开启 align_depth.enable:=true）
ros2 launch dog_follow follow.launch.py range:=depth depth:=/camera/aligned_depth_to_color/image_raw
# ZED 2i：深度图是 float32 米
ros2 launch dog_follow follow.launch.py range:=depth image:=/zed/zed_node/rgb/image_rect_color \
    camera_info:=/zed/zed_node/rgb/camera_info depth:=/zed/zed_node/depth/depth_registered
```
`DepthImageEstimator` 取检测框中上部区域深度的第 30 百分位作为距离。之后还可以：直接使用 ZED 的 body_tracking（自带 3D 坐标和 ID），以及接入 `nav2_collision_monitor` 做避障。

## 8. 测试与离线调参
```bash
python -m pytest src/dog_follow/test -q           # 核心算法单测（不依赖 ROS）
python -m dog_follow.tools.offline_demo --video walk.mp4 --det yolox_s.onnx --reid osnet_x0_25.onnx --camera-height 0.45
```
建议用狗身上的相机录几段视频（含直行、转弯、遮挡、多人交叉），先离线把 ID 切换和测距误差调好，再上实机。
