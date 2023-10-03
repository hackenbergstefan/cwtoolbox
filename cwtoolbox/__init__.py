# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

import contextlib
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

import numpy as np
import numpy.typing
import tqdm


class CaptureDevice:
    def compile(
        self,
        file: Optional[Union[str, Path]] = None,
        code: Optional[str] = None,
        cflags: str = "",
        ldflags: str = "",
    ):
        """Compile given file or code string."""

    def flash(self, file: Optional[str] = None):
        """Flash compiled code to device."""

    def reset_target(self):
        """Reset underlying device."""

    def connect(self):
        """Connect underlying device."""

    def disconnect(self):
        """Disconnect underlying device."""

    def capture(
        self,
        number_of_traces: int,
        number_of_samples: int,
        input: Callable[[int], List[int]],
    ) -> numpy.typing.NDArray:
        """Capture traces."""
        data = np.empty(
            number_of_traces,
            dtype=[
                ("trace", "f8", (number_of_samples,)),
                ("input", "u1", (len(input(0)),)),
            ],
        )
        with self.connected():
            self.reset_target()
            for i in tqdm.tqdm(range(number_of_traces)):
                data["input"][i, :] = input(i)
                data["trace"][i, :] = self.capture_single_trace(
                    number_of_samples,
                    data["input"][i, :],
                )
        return data

    def capture_single_trace(
        self,
        number_of_samples: int,
        input: Iterable[int],
    ) -> numpy.typing.NDArray:
        """Capture one single traces."""
        list(input)
        return np.zeros(number_of_samples)

    @contextlib.contextmanager
    def connected(self):
        """Decorator to execute code in connected state."""
        try:
            self.connect()
            yield
        finally:
            self.disconnect()

    @staticmethod
    def create(platform, **kwargs) -> "CaptureDevice":
        if platform.startswith("CW"):
            # pylint: disable=import-outside-toplevel
            from .cw import CwCaptureSimpleSerial

            return CwCaptureSimpleSerial(platform)
        if platform.startswith("XMLRPCCW"):
            platform = platform.removeprefix("XMLRPC")
            # pylint: disable=import-outside-toplevel
            from .cw import (
                CwCaptureSimpleSerialRpcClient,
                CwCaptureSimpleSerialRpcClientResourceServer,
            )

            if "resource_server_uri" in kwargs:
                return CwCaptureSimpleSerialRpcClientResourceServer(
                    platform,
                    **kwargs,
                )

            return CwCaptureSimpleSerialRpcClient(platform, **kwargs)

        raise NotImplementedError(f"Unsupported Platform: {platform}")
