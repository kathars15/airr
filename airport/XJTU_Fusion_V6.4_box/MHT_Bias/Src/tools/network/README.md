# network 目录说明

这里放联机网络辅助脚本，核心用途是：

- 雷达 UDP 分发
- 光电 UDP 分发
- 上位机软件与 `main2.py` 同时收数
- 当前监听状态检查
- 光电 RTSP 网络连通性检查

## 推荐入口

- `udp_fanout.py`
  - UDP 分发器主脚本。

- `check_udp_fanout_status.py`
  - 检查分发器、监听端口、收包情况。

- `configure_eosmsv4_for_fanout.py`
  - 辅助检查或适配上位机接收配置。

- `run_full_system.ps1`
  - 一键拉起完整链路的 PowerShell 脚本。

- `run_main2_with_fanout_ports.ps1`
  - 用分发端口启动 `main2.py`。

- `run_udp_fanout.ps1`
  - 单独启动分发器。

- `check_optical_rtsp_network.py`
  - 检查本机是否处于 `192.168.0.x` 网段。
  - 同时检查 `192.168.0.98:554` 的 `ping` 和 TCP 连通性。
  - 用来快速判断“RTSP 不通到底是代码问题还是网络问题”。

- `set_optical_network_static_ip.ps1`
  - 给指定网卡设置静态 IPv4。
  - 默认只预览，不改机器。
  - 只有显式加 `-Apply` 才会真的执行。

## 当前建议

如果目的是“雷达软件 + main2 同时收包”，优先看：

1. `udp_fanout.py`
2. `check_udp_fanout_status.py`
3. `run_full_system.ps1`

如果目的是“在线识别窗口为什么收不到 RTSP”，优先看：

1. `check_optical_rtsp_network.py`
2. `CV/code_image/test_rtsp_gpu.py`
3. `set_optical_network_static_ip.ps1`

## 在线识别链路排查顺序

### 1. 先查网络

```powershell
python check_optical_rtsp_network.py
```

若结果显示：

```text
local_machine_not_in_192.168.0.x_subnet
```

说明本机根本没连到光电设备所在网段，先切网卡/IP，再谈 RTSP。

### 2. 再查 RTSP + GPU + 模型

```powershell
python D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\CV\code_image\test_rtsp_gpu.py
```

它会打印：

- 当前模型权重路径
- RTSP 地址
- `torch`
- `cuda_available`
- `cuda_device`
- `predict_device`
- RTSP 是否能打开
- 第一帧是否能读到
- 单帧推理耗时

### 3. 最后再开完整主程序

等 1 和 2 都通过后，再启动：

- 分发器
- `main2.py`
- 在线识别窗口

## 安全设置静态 IP

先只预览可用网卡：

```powershell
.\set_optical_network_static_ip.ps1
```

再预览某张网卡的改动计划：

```powershell
.\set_optical_network_static_ip.ps1 -InterfaceAlias "Ethernet" -IpAddress 192.168.0.9
```

确认无误后，使用管理员 PowerShell 执行：

```powershell
.\set_optical_network_static_ip.ps1 -InterfaceAlias "Ethernet" -IpAddress 192.168.0.9 -Apply
```
