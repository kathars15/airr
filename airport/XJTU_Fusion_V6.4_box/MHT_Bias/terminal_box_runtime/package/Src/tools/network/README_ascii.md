# network quick guide

This file keeps only ASCII text so Windows terminal will not show mojibake.

## Main scripts

- `udp_fanout.py`
- `check_udp_fanout_status.py`
- `check_optical_rtsp_network.py`
- `set_optical_network_static_ip.ps1`
- `run_full_system.ps1`

## Optical RTSP check order

1. Check network:

```powershell
python check_optical_rtsp_network.py
```

2. Check RTSP + GPU + model:

```powershell
python D:\desk\airr\airport\XJTU_Fusion_V6.4_box\MHT_Bias\CV\code_image\test_rtsp_gpu.py
```

3. Then start full system.

## Safe static IP helper

Preview only:

```powershell
.\set_optical_network_static_ip.ps1
```

Preview one NIC:

```powershell
.\set_optical_network_static_ip.ps1 -InterfaceAlias "Ethernet" -IpAddress 192.168.0.9
```

Really apply only with:

```powershell
.\set_optical_network_static_ip.ps1 -InterfaceAlias "Ethernet" -IpAddress 192.168.0.9 -Apply
```

## Common result meanings

- `local_machine_not_in_192.168.0.x_subnet`
  - Your PC is not on same subnet as optical device.
  - Switch to wired NIC or set one NIC to `192.168.0.x`.

- `tcp_failed(192.168.0.98:554)`
  - RTSP port not reachable.

- `opened: False`
  - OpenCV could not open RTSP stream.
