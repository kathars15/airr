# 光电网络切换说明

目标设备当前默认地址：

- 光电控制/上报：`192.168.0.98`
- RTSP：`192.168.0.98:554`

如果本机不在 `192.168.0.x` 网段，RTSP 与光电 UDP 通常都不会通。

## 一、先判断当前是否在正确网段

运行：

```powershell
& D:/anaconda/python.exe D:/desk/airr/airport/XJTU_Fusion_V6.4_box/MHT_Bias/Src/tools/network/check_optical_rtsp_network.py
```

如果看到：

```text
local_machine_not_in_192.168.0.x_subnet
```

就说明要先切网络。

## 二、推荐做法

### 做法 1：切到现场有线网卡

如果你的雷达/光电现场链路本来就在有线网口上：

1. 插上现场网线
2. 关闭或断开当前 `WLAN`
3. 确认有线网卡拿到 `192.168.0.x` 地址，或手动配置
4. 再运行 `check_optical_rtsp_network.py`

这是最推荐方式。

### 做法 2：给指定网卡手动设静态 IP

假设你要把某张网卡设为：

- IP：`192.168.0.9`
- 掩码：`255.255.255.0`

Windows 图形界面做法：

1. 打开“网络和 Internet 设置”
2. 找到对应网卡
3. 进入 IPv4 属性
4. 手动设置：
   - `IP地址：192.168.0.9`
   - `子网掩码：255.255.255.0`
   - 网关可先留空

### 做法 2A：使用脚本安全预览/应用

先预览：

```powershell
.\set_optical_network_static_ip.ps1
```

再预览指定网卡：

```powershell
.\set_optical_network_static_ip.ps1 -InterfaceAlias "Ethernet" -IpAddress 192.168.0.9
```

最后只在确认无误且使用管理员 PowerShell 时执行：

```powershell
.\set_optical_network_static_ip.ps1 -InterfaceAlias "Ethernet" -IpAddress 192.168.0.9 -Apply
```

### 做法 3：PowerShell 临时查看网卡

```powershell
Get-NetIPAddress -AddressFamily IPv4
```

重点看有没有某张网卡处于：

```text
192.168.0.x/24
```

## 三、切完后怎么复验

### 1. 先看网络层

```powershell
python check_optical_rtsp_network.py
```

理想情况：

- `same_subnet_with_192.168.0.98=yes`
- `ping success=yes` 或至少不再超时
- `tcp_ok(192.168.0.98:554)`

### 2. 再看 RTSP + GPU

```powershell
python D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\CV\code_image\test_rtsp_gpu.py
```

理想情况：

- `opened: True`
- `first_frame_ok: True`
- `predict_device: 0`

## 四、若仍然不通

如果已经在 `192.168.0.x` 网段，但还是不通，优先检查：

1. 光电设备是否上电
2. 光电设备网线/交换机是否正常
3. 光电设备 RTSP 是否确实开在 `554`
4. 当前通道是否为 `channel=0,stream=0`
5. 设备是否限制单客户端占流

## 五、当前结论

你之前的诊断结果表明：

- 本机只有 `WLAN 10.136.237.241/24`
- 不在 `192.168.0.x`

所以那时 RTSP 不通是正常现象，不是模型或 GPU 问题。
