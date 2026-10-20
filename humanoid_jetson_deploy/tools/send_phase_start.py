#!/usr/bin/env python3
"""Send one local UDP start cue after the 8 cm toe-to-stick placement is ready."""

import argparse
import ipaddress
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5008)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.host)
    if address.version != 4 or not address.is_loopback or not 1 <= args.port <= 65535:
        parser.error("host must be an IPv4 loopback address and port must be 1..65535")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(b'{"start": true}', (args.host, args.port))
    print(f"Sent one phase-clock start cue to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
