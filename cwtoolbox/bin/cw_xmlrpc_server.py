#!/usr/bin/env python

# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT


"""Start XML-RPC Server."""

import argparse
import logging
import xmlrpc
import xmlrpc.server

import cwtoolbox.cw

logging.basicConfig(level="debug")


def main():
    """Start XML-RPC Server."""
    parser = argparse.ArgumentParser(
        "ChipWhisperer XMLRPC Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--address",
        default="0.0.0.0",
        help="Address to export server to.",
    )
    parser.add_argument(
        "-p",
        "--port",
        default="8000",
        type=int,
        help="Port number for server.",
    )
    args = parser.parse_args()

    with xmlrpc.server.SimpleXMLRPCServer(
        (args.address, args.port),
        allow_none=True,
        use_builtin_types=True,
    ) as server:
        server.register_instance(
            cwtoolbox.cw.CwCaptureSimpleSerialRpcService(),
            allow_dotted_names=True,
        )
        server.register_introspection_functions()
        server.serve_forever()


if __name__ == "__main__":
    main()
