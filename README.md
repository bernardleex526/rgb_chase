# rgb_chase — 机器狗 RGB 人体跟随

与机器狗型号无关的 RGB 相机人体跟随（ROS 2 Humble/Jazzy，目标平台 Jetson Orin）。
第一阶段使用单目相机，第二阶段支持深度相机（ZED / D435i）。

```
相机 → follow_perception（YOLOX → ByteTrack → OSNet ReID 锁定 → 测距）
     → follow_controller（PD + 限速/限加速度）
     → 适配节点（m20_adapter / twist_adapter）→ 机器狗
```

| 路径 | 内容 |
|---|---|
| [`src/dog_follow/`](src/dog_follow/) | ROS 2 包，完整说明见 [`src/dog_follow/README.md`](src/dog_follow/README.md) |
| `src/dog_follow/dog_follow/core/` | 纯算法（不依赖 ROS，可在 PC 上直接用 pytest 测试） |
| `src/dog_follow/dog_follow/backends/` | ONNXRuntime / TensorRT 推理后端 |
| `src/dog_follow/dog_follow/tools/offline_demo.py` | 不需要 ROS 的离线调参工具 |

快速开始：

```bash
colcon build --symlink-install && source install/setup.bash
python -m pytest src/dog_follow/test -q        # 核心算法单测
ros2 launch dog_follow follow.launch.py robot:=none image:=/camera/color/image_raw camera_info:=/camera/color/camera_info
```

控制器默认不输出速度，需要显式调用 `/follow_controller/enable`。M20 适配走官方 `sdk_deploy` 的 ROS 2 指令模式，
首次上机前请先读子目录 README 第 5 节的安全说明。

相关仓库：3D 点云跟随见 [`m20_chase`](https://github.com/bernardleex526/m20_chase)，2D 雷达跟随与操作员网关见 [`deamon2`](https://github.com/bernardleex526/deamon2)。
