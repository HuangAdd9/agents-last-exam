#!/usr/bin/env python3
"""Manage one shared IPv4/TCP proxy network for local QEMU runners (Linux/systemd)."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

NETWORK = "ale-qemu-egress"
BRIDGE = "br-ale-egress"
CHAIN = "ALE_QEMU_EGRESS"
LABEL = "io.agents-last-exam.egress"
UNIT = "ale-qemu-egress.service"
STATE = Path("/run/ale-qemu-egress")
PORT = 17890
# 只处理专用网桥进入的流量，不修改宿主机 OUTPUT 或默认路由。
NAT_HOOK = ["-i", BRIDGE, "-p", "tcp", "-j", CHAIN]
QUIC_RULE = ["-i", BRIDGE, "-p", "udp", "--dport", "443", "-j", "REJECT"]


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, text=True, capture_output=True, timeout=30, check=False)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result


def inspect_network() -> dict | None:
    result = run("docker", "network", "inspect", NETWORK, check=False)
    if result.returncode:
        if "not found" in result.stderr.lower() or "no such network" in result.stderr.lower():
            return None
        raise RuntimeError(result.stderr)
    network = json.loads(result.stdout)[0]
    # 同名但不是本脚本创建的网络，不能擅自接管或删除。
    if (
        (network.get("Labels") or {}).get(LABEL) != "1"
        or network["Driver"] != "bridge"
        or network["Options"].get("com.docker.network.bridge.name") != BRIDGE
        or network.get("EnableIPv6")
    ):
        raise RuntimeError(f"{NETWORK} exists but is not this script's IPv4 bridge")
    return network


def healthy() -> bool:
    service = run("systemctl", "is-active", "--quiet", UNIT, check=False)
    hook = run("iptables", "-w", "-t", "nat", "-C", "PREROUTING", *NAT_HOOK, check=False)
    quic = run("iptables", "-w", "-C", "DOCKER-USER", *QUIC_RULE, check=False)
    return all(result.returncode == 0 for result in (service, hook, quic))


def stop() -> None:
    network = inspect_network()
    if network:
        if network.get("Containers"):
            raise RuntimeError("Network still has containers; leave their gateway running")
        # 先删除网络：Docker 会原子检查是否还有容器，避免检查后恰有新 VM 加入。
        run("docker", "network", "rm", NETWORK)
    for table, parent, rule in (
        ("nat", "PREROUTING", NAT_HOOK),
        ("filter", "DOCKER-USER", QUIC_RULE),
    ):
        if run("iptables", "-w", "-t", table, "-C", parent, *rule, check=False).returncode == 0:
            run("iptables", "-w", "-t", table, "-D", parent, *rule)
    if run("iptables", "-w", "-t", "nat", "-S", CHAIN, check=False).returncode == 0:
        run("iptables", "-w", "-t", "nat", "-F", CHAIN)
        run("iptables", "-w", "-t", "nat", "-X", CHAIN)
    result = run("systemctl", "stop", UNIT, check=False)
    if result.returncode and "not loaded" not in result.stderr:
        raise RuntimeError(result.stderr)


def start(binary: str, direct_domains: list[str]) -> None:
    network = inspect_network()
    if network and healthy():
        print(f"{NETWORK} already running; existing configuration retained")
        return
    if network and network.get("Containers"):
        raise RuntimeError("Gateway is unhealthy but containers remain; inspect before changing it")
    executable = shutil.which(binary)
    if not executable:
        raise RuntimeError("Pass --mihomo /absolute/path/to/mihomo")
    executable = str(Path(executable).resolve())
    # 7890 必须支持 SOCKS5；只检查本地握手，不发送凭据。
    with socket.create_connection(("127.0.0.1", 7890), timeout=3) as upstream:
        upstream.sendall(b"\x05\x01\x00")
        reply = upstream.makefile("rb").read(2)
        if reply != b"\x05\x00":
            raise RuntimeError("127.0.0.1:7890 must accept unauthenticated SOCKS5")
    stop()  # 没有 VM 时，清理上一次失败启动留下的规则。
    try:
        run(
            "docker",
            "network",
            "create",
            "--driver",
            "bridge",
            "--ipv6=false",
            "--label",
            f"{LABEL}=1",
            "--opt",
            f"com.docker.network.bridge.name={BRIDGE}",
            NETWORK,
        )
        network = inspect_network()
        gateway = network["IPAM"]["Config"][0]["Gateway"]
        STATE.mkdir(mode=0o755, exist_ok=True)
        # 第二个 mihomo 只做转接，不复制现有订阅、密钥或全局配置。
        config = {
            "redir-port": PORT,
            "bind-address": gateway,
            "allow-lan": True,
            "ipv6": False,
            "mode": "rule",
            "log-level": "info",
            "proxies": [
                {"name": "host-proxy", "type": "socks5", "server": "127.0.0.1", "port": 7890}
            ],
            "sniffer": {
                "enable": True,
                "parse-pure-ip": True,
                "sniff": {"TLS": {"ports": [443]}, "HTTP": {"ports": [80, 8080]}},
            },
            # 拒绝以转接器自身为目标的连接，防止误探测透明端口形成回路。
            "rules": [f"IP-CIDR,{gateway}/32,REJECT,no-resolve"]
            + [f"DOMAIN-SUFFIX,{domain},DIRECT" for domain in direct_domains]
            + ["MATCH,host-proxy"],
        }
        path = STATE / "config.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        run(executable, "-t", "-d", str(STATE), "-f", str(path))
        run(
            "systemd-run",
            "--unit",
            UNIT,
            "--collect",
            "--property=Restart=on-failure",
            executable,
            "-d",
            str(STATE),
            "-f",
            str(path),
        )
        for attempt in range(50):
            # 透明端口必须通过 NAT 使用；健康检查只读监听状态，不直接连接它。
            listeners = run("ss", "-H", "-lnt", f"sport = :{PORT}").stdout
            if any(line.split()[3] == f"{gateway}:{PORT}" for line in listeners.splitlines()):
                break
            if attempt == 49:
                raise RuntimeError(f"Relay did not start; check journalctl -u {UNIT}")
            time.sleep(0.1)

        run("iptables", "-w", "-t", "nat", "-N", CHAIN)
        # 本机、内网和共享目录直接连接；公网 TCP 才进入转接器。
        run(
            "iptables",
            "-w",
            "-t",
            "nat",
            "-A",
            CHAIN,
            "-m",
            "addrtype",
            "--dst-type",
            "LOCAL",
            "-j",
            "RETURN",
        )
        for cidr in (
            "0.0.0.0/8",
            "10.0.0.0/8",
            "100.64.0.0/10",
            "127.0.0.0/8",
            "169.254.0.0/16",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "224.0.0.0/4",
        ):
            run("iptables", "-w", "-t", "nat", "-A", CHAIN, "-d", cidr, "-j", "RETURN")
        run(
            "iptables",
            "-w",
            "-t",
            "nat",
            "-A",
            CHAIN,
            "-p",
            "tcp",
            "-j",
            "REDIRECT",
            "--to-ports",
            str(PORT),
        )
        # 最小实现只代理 TCP；拒绝 QUIC，让支持回退的 HTTPS 客户端改用 TCP。
        run("iptables", "-w", "-I", "DOCKER-USER", "1", *QUIC_RULE)
        run("iptables", "-w", "-t", "nat", "-I", "PREROUTING", "1", *NAT_HOOK)
        print(f"Ready: {NETWORK} -> {gateway}:{PORT} -> 127.0.0.1:7890")
    except BaseException:
        stop()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--mihomo", default="mihomo", help="existing mihomo executable")
    parser.add_argument("--direct-domain", action="append", default=["maas.yicloud.com.cn"])
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("run with sudo (Docker networking and iptables require root)")
    # 多个终端操作时串行执行；任务容器的接入/退出由 Docker 自己管理。
    with Path("/run/ale-qemu-egress.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == "start":
            start(args.mihomo, args.direct_domain)
        elif args.action == "stop":
            stop()
        else:
            network = inspect_network()
            if not network or not healthy():
                raise SystemExit("Gateway not ready; inspect journalctl -u " + UNIT)
            print(
                json.dumps(
                    {"network": NETWORK, "containers": network.get("Containers", {})}, indent=2
                )
            )
            print(run("iptables", "-w", "-t", "nat", "-L", CHAIN, "-n", "-v").stdout)


if __name__ == "__main__":
    main()
