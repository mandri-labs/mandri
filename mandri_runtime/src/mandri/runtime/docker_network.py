import ipaddress
import socket

import psutil
from mandri.runtime.errors.docker import DockerExecutionError

DENIED_NETWORKS = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "224.0.0.0/4",
    "240.0.0.0/4",
    "::/128",
    "::1/128",
    "::ffff:0:0/96",
    "64:ff9b:1::/48",
    "100::/64",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
)


def local_networks() -> tuple[str, ...]:
    networks = set(DENIED_NETWORKS)
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family not in (socket.AF_INET, socket.AF_INET6) or not address.netmask:
                continue
            value = address.address.split("%", 1)[0]
            try:
                networks.add(str(ipaddress.ip_network(f"{value}/{address.netmask}", strict=False)))
            except ValueError:
                networks.add(str(ipaddress.ip_network(value)))
    return tuple(sorted(networks))


def network_exception(value: str) -> tuple[str, int]:
    try:
        address, raw_port = value.rsplit(":", 1)
        network = ipaddress.ip_network(address.strip("[]"), strict=False)
        port = int(raw_port)
        if not 1 <= port <= 65535 or network.prefixlen == 0:
            raise ValueError
    except (ValueError, TypeError) as error:
        raise DockerExecutionError(
            "docker_network_policy_invalid",
            "Network exceptions require a literal address and TCP port",
        ) from error
    return str(network), port
