# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

"""Capture Rainbow-rs."""

import json
import logging
import multiprocessing
import socket
import struct
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import tqdm
from chipwhisperer.capture.targets.SimpleSerial2 import SimpleSerial2
from numpy._typing import NDArray

from ..cw import CwCaptureSimpleSerial

GITHUB_URL = "https://github.com/hackenbergstefan/rainbow-rs/releases/download/"


def download_and_extract(rainbow_version="v0.6.0"):
    """Download Rainbow from github release."""
    if sys.platform == "linux":
        package_name = "rainbow-rs-x86_64-unknown-linux-gnu.tar.gz"
    elif sys.platform == "win32":
        package_name = "rainbow-rs-x86_64-pc-windows-msvc.zip"
    else:
        raise FileNotFoundError(f"No prebuilt package for {sys.platform}")
    url = f"{GITHUB_URL}/{rainbow_version}/{package_name}"
    file = (Path(__file__).parent / package_name).absolute()

    logging.getLogger(__name__).info(f"Downloading Rainbow from {url}")
    urllib.request.urlretrieve(url, file)

    if package_name.endswith(".tar.gz"):
        with tarfile.open(file) as tar:
            tar.extractall(Path(__file__).parent)
    elif package_name.endswith(".zip"):
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(Path(__file__).parent)
    file.unlink()
    rainbow = Path(__file__).parent / "rainbow-rs"
    assert rainbow.exists()


def free_port() -> int:
    """Return a free socket port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


class RainbowSimpleSerialSocket(socket.socket):
    # pylint: disable=abstract-method,super-init-not-called
    """SimpleSerial Socket interface for Rainbow."""

    def __init__(self, port, timeout=10.0):
        socket.socket.__init__(self, socket.AF_INET, socket.SOCK_STREAM)
        self.settimeout(timeout)
        for _ in range(10):
            try:
                self.connect(("127.0.0.1", port))
                break
            except OSError:
                time.sleep(0.5)
        self.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.simpleserial = SimpleSerial2()

    def simpleserial_payload(self, cmd, data):
        """Generate payload data for communication with victim."""
        if isinstance(data, list):
            data = bytearray(data)
        if isinstance(cmd, str):
            cmd = ord(cmd[0])
        buf = [0x00, cmd, 0x00, len(data)]
        buf.extend(data)
        crc = self.simpleserial._calc_crc(buf[1:])  # pylint: disable=protected-access
        buf.append(crc)
        buf.append(0x00)
        buf = self.simpleserial._stuff_data(buf)  # pylint: disable=protected-access
        return buf

    def write(self, data: bytes):
        """Send given bytes to Rainbow."""
        raise NotImplementedError("")

    def read_line(self) -> bytes:
        r"""Read one line terminated by `\n` from the socket."""
        data = b""
        while not data or data[-1:] != b"\n":
            data += self.recv(1024 * 1024)
        return data

    def generate_trace(
        self,
        idx,
        cmd,
        data,
    ) -> Tuple[List[float], List[str]]:
        """Generate a trace with instructions."""
        data = self.simpleserial_payload(cmd, data)
        self.sendall(json.dumps({"GetTraceWithInstructions": (idx, data)}).encode() + b"\n")
        response = json.loads(self.read_line())["TraceWithInstructions"]
        return (
            response[1],
            response[2],
        )

    def generate_trace_async(self, idx, cmd, data):
        """Generate a trace async."""
        data = self.simpleserial_payload(cmd, data)
        self.sendall(json.dumps({"VictimData": (idx, data)}).encode() + b"\n")

    def get_trace_binary(self, number_of_samples: int) -> Tuple[int, NDArray]:
        """Send `GetTraceBinary` and receive answer."""
        self.sendall(b'"GetTraceBinary"\n')
        buf = b""
        while len(buf) < 4 + 4 * number_of_samples:
            buf += self.recv(4 + 4 * number_of_samples - len(buf))
        trace = struct.unpack(f"!I{number_of_samples}f", buf)
        return trace[0], np.array(trace[1:])

    def __del__(self):
        """Override __del__."""


class RainbowCwCaptureSimpleSerial(CwCaptureSimpleSerial):
    """CaptureDevice for Rainbow-rs."""

    def __init__(
        self,
        cw_platform: str = "CWLITEARM",
        rainbow_exe: Optional[Union[str, Path]] = None,
        rainbow_args: Optional[Dict[str, Optional[str]]] = None,
    ):
        if not rainbow_exe:
            rainbow_exe = Path(__file__).parent / "rainbow-rs"
            if not rainbow_exe.exists():
                download_and_extract()
        self.rainbow_exe = Path(rainbow_exe)
        self.rainbow_args = rainbow_args or {
            "threads": str(multiprocessing.cpu_count()),
        }
        self.rainbow: Optional[subprocess.Popen] = None
        self.rainbow_sock: Optional[RainbowSimpleSerialSocket] = None
        self.rainbow_sock_timeout: Optional[float] = 10.0
        self.instruction_trace: Optional[List[str]] = None
        super().__init__(cw_platform)

    def connect(self, rainbow_args: Optional[Dict[str, Optional[str]]] = None):
        """Start and connect to Rainbow-rs."""
        port = free_port()
        args = self.rainbow_args.copy()
        args.update(rainbow_args or {})
        args["socket"] = f"127.0.0.1:{port}"
        args["memory-extension"] = args.get("memory-extension", "no-bus-no-cache")
        args["leakage"] = args.get("leakage", "elmo")
        if args["leakage"] == "elmo":
            args["coefficientfile"] = args.get(
                "coefficientfile",
                str((Path(__file__).parent / "coeffs.txt").absolute()),
            )

        self.rainbow = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                self.rainbow_exe.absolute(),
                self.buildfolder / f"generic_simpleserial-{self.cw_platform}.elf",
            ]
            + sum(([f"--{k}", v] if v else [f"--{k}"] for k, v in args.items()), [])
        )
        self.rainbow_sock = RainbowSimpleSerialSocket(port=port, timeout=self.rainbow_sock_timeout)

    def disconnect(self):
        """Disconnect from Rainbow-rs."""
        if self.rainbow_sock:
            self.rainbow_sock.close()
            self.rainbow_sock = None

        if self.rainbow:
            self.rainbow.terminate()
            self.rainbow.wait()
            self.rainbow = None

    def reset_target(self):
        """Override."""

    def flash(self, _file: str | None = None):
        """Override."""

    def capture_single_trace(
        self,
        number_of_samples: int,
        input: Union[Iterable[int], bytes],
        read_output: bool = False,
    ) -> Tuple[NDArray, Optional[bytes]]:
        """
        Capture one single trace.

        Parameters
        ----------
        number_of_samples
            The number of samples to capture.
        input
            A list of input values for each trace.
        read_output
            Whether to return the output generated by the device.

        Returns
        -------
        numpy.ndarray
            A trace.

        """
        assert self.rainbow_sock is not None

        trace, self.instruction_trace = self.rainbow_sock.generate_trace(
            idx=0,
            cmd=0x01,
            data=bytes(input),
        )
        return np.array(trace), None

    def capture_yield(
        self,
        number_of_traces: int,
        input: dict[str, Callable[[int], Union[bytes, List[int]]]],
        number_of_samples: int = 0,
    ) -> Iterable[Tuple[int, dict[str, List[int]], NDArray]]:
        """
        Ram-saving version of capture yielding the data instead of returning.

        Parameters
        ----------
        number_of_traces
            The number of traces to capture.
        input
            The dictionary where each key is a string representing the name of the input field,
            and each value is a function that accepts an integer (the index) and returns
            a byte string or list of integers.
            This function is applied to generate the input field data.
        number_of_samples
            The number of samples to capture for each trace.

        Returns
        -------
        int
            Index of current trace.
        List[int]
            Input for this index.
        numpy.ndarray
            Array of one trace.
        """
        assert number_of_samples != 0

        with self.connected():
            assert self.rainbow_sock is not None

            inputs = []
            for i in tqdm.tqdm(range(number_of_traces)):
                inputs.append({name: list(func(i)) for name, func in input.items()})
                self.rainbow_sock.generate_trace_async(
                    i,
                    0x01,
                    b"".join(map(bytes, inputs[i].values())),
                )

            for _ in tqdm.tqdm(range(number_of_traces)):
                idx, trace = self.rainbow_sock.get_trace_binary(number_of_samples)
                yield idx, inputs[idx], trace
