import ipaddress
import socket
import subprocess
from pathlib import Path


OPTICAL_IP = "192.168.0.98"
OPTICAL_PORT = 554


def run_ps(command: str) -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    return (result.stdout or "") + (result.stderr or "")


def tcp_check(host: str, port: int, timeout_sec: float = 2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True, f"tcp_ok({host}:{port})"
    except Exception as exc:
        return False, f"tcp_failed({host}:{port}): {exc}"


def ping_check(host: str):
    result = subprocess.run(
        ["ping", host, "-n", "2"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    return result.returncode == 0, result.stdout


def list_ipv4_interfaces():
    cmd = (
        "Get-NetIPAddress -AddressFamily IPv4 | "
        "Where-Object { $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1' } | "
        "Select-Object InterfaceAlias,IPAddress,PrefixLength | ConvertTo-Json -Compress"
    )
    raw = run_ps(cmd).strip()
    if not raw:
        return []
    import json

    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    return data


def same_subnet(ip_text: str, prefix_len: int, target_ip: str):
    try:
        iface = ipaddress.ip_interface(f"{ip_text}/{prefix_len}")
        target = ipaddress.ip_address(target_ip)
    except ValueError:
        return False
    return target in iface.network


def main():
    print("=== Optical RTSP Network Check ===")
    print(f"target_rtsp: rtsp://{OPTICAL_IP}:{OPTICAL_PORT}/channel=0,stream=0")
    print()

    interfaces = list_ipv4_interfaces()
    print("[local_ipv4_interfaces]")
    if not interfaces:
        print("  no_ipv4_interface_found")
    else:
        for item in interfaces:
            alias = item.get("InterfaceAlias")
            ip_addr = item.get("IPAddress")
            prefix = int(item.get("PrefixLength", 0))
            in_same = same_subnet(ip_addr, prefix, OPTICAL_IP)
            print(
                f"  alias={alias} ip={ip_addr}/{prefix} "
                f"same_subnet_with_{OPTICAL_IP}={'yes' if in_same else 'no'}"
            )
    print()

    ping_ok, ping_output = ping_check(OPTICAL_IP)
    print("[ping_check]")
    print(f"  success={'yes' if ping_ok else 'no'}")
    for line in ping_output.strip().splitlines():
        print(f"  {line}")
    print()

    tcp_ok, tcp_reason = tcp_check(OPTICAL_IP, OPTICAL_PORT)
    print("[tcp_check]")
    print(f"  {tcp_reason}")
    print()

    same_subnet_any = any(
        same_subnet(item.get("IPAddress", ""), int(item.get("PrefixLength", 0)), OPTICAL_IP)
        for item in interfaces
    )
    print("[summary]")
    if not same_subnet_any:
        print(f"  local_machine_not_in_{OPTICAL_IP.rsplit('.', 1)[0]}.x_subnet")
        print("  action: switch to optical/radar wired NIC or set a NIC to 192.168.0.x")
    elif not ping_ok:
        print("  subnet_may_be_correct_but_device_not_replying_ping")
        print("  action: check device power, cable, switch, or ICMP policy")
    elif not tcp_ok:
        print("  device_reachable_but_rtsp_port_closed_or_blocked")
        print("  action: check RTSP service/channel/port 554")
    else:
        print("  network_and_rtsp_port_look_ok")


if __name__ == "__main__":
    main()
