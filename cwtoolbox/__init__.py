# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

"""Cwtoolbox unifies access to ChipWhisperer®-Devices and other SCA trace sources."""

import contextlib
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import lascar
import numpy as np
import numpy.typing
import tqdm


def input_dict_to_array(
    number_of_traces: int,
    number_of_samples: int,
    input: dict[str, Callable[[int], Union[bytes, List[int]]]],
    additional_fields: dict[
        str,
        Callable[[numpy.typing.NDArray], Union[bytes, List[int]]],
    ],
) -> numpy.typing.NDArray:
    """Convert dict of input functions to numpy array of fitting size."""
    additional_fields = additional_fields or {}

    if number_of_samples == 0:
        dtype = np.dtype([])
    else:
        dtype = np.dtype([("trace", "f8", (number_of_samples,))])
    dtype = np.dtype(dtype.descr + [(name, "u1", (len(func(0)),)) for name, func in input.items()])
    data = np.empty(number_of_traces, dtype=dtype)
    for name, func in additional_fields.items():
        dtype = np.dtype(dtype.descr + [(name, "u4", np.array(func(data[0])).shape)])
        data = np.empty(number_of_traces, dtype=dtype)
    return data


class CaptureDevice:
    """A class used to capture traces from an external device."""

    def compile(
        self,
        file: Optional[Union[str, Path]] = None,
        code: Optional[str] = None,
        cflags: str = "",
        ldflags: str = "",
    ) -> None:
        """
        Compile given file or code string.

        Parameters
        ----------
        file
            The file to be compiled.
        code
            The code string to be compiled.
        cflags
            Compilation flags.
        ldflags
            Linking flags.

        Returns
        -------
        None

        """

    def flash(self, file: Optional[str] = None) -> None:
        """
        Flash compiled code to device.

        Parameters
        ----------
        file
            The file containing the compiled code to be flashed.

        Returns
        -------
        None

        """

    def reset_target(self) -> None:
        """
        Reset underlying device.

        Returns
        -------
        None

        """

    def connect(self) -> None:
        """
        Connect underlying device.

        Returns
        -------
        None

        """

    def disconnect(self) -> None:
        """
        Disconnect underlying device.

        Returns
        -------
        None

        """

    def capture(
        self,
        number_of_traces: int,
        input: dict[str, Callable[[int], Union[bytes, List[int]]]],
        additional_fields: Optional[
            dict[str, Callable[[numpy.typing.NDArray], Union[bytes, List[int]]]]
        ] = None,
        number_of_samples: int = 0,
        sample_range: Optional[Tuple[int, int]] = None,
    ) -> numpy.typing.NDArray:
        """
        Capture traces.

        Parameters
        ----------
        number_of_traces
            The number of traces to capture.
        input
            The dictionary where each key is a string representing the name of the input field,
            and each value is a function that accepts an integer (the index) and returns
            a byte string or list of integers.
            This function is applied to generate the input field data.
        additional_fields
            The dictionary where each key is a string representing the name of an additional field,
            and each value is a function that takes the current trace and returns
            a byte string or list of integers.
            This function is applied to generate the additional field data.
        number_of_samples
            The number of samples to capture for each trace.
        sample_range
            If not None, only the range of these samples are is used of each trace.

        Returns
        -------
        numpy.ndarray
            Structured array of traces where each trace contains an input array and a trace array.

        """
        additional_fields = additional_fields or {}
        if number_of_samples == 0:
            with self.connected():
                self.reset_target()
                trace, _ = self.capture_single_trace(
                    input=b"".join(bytes(func(0)) for func in input.values()),
                    number_of_samples=0,
                )
                number_of_samples = len(trace)

        number_of_samples_cropped = (
            (sample_range[1] - sample_range[0]) if sample_range else number_of_samples
        )
        data = input_dict_to_array(
            number_of_traces=number_of_traces,
            number_of_samples=number_of_samples_cropped,
            input=input,
            additional_fields=additional_fields,
        )
        for i, indata, trace in self.capture_yield(
            number_of_traces=number_of_traces,
            number_of_samples=number_of_samples,
            input=input,
        ):
            data["trace"][i, :] = (
                trace[sample_range[0] : sample_range[1]] if sample_range else trace
            )
            for name, value in indata.items():
                data[name][i, :] = value
            for name, func in additional_fields.items():
                data[name][i, :] = func(data[i])
        return data

    def capture_yield(
        self,
        number_of_traces: int,
        input: dict[str, Callable[[int], Union[bytes, List[int]]]],
        number_of_samples: int = 0,
    ) -> Iterable[Tuple[int, dict[str, List[int]], numpy.typing.NDArray]]:
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
        List[dict[str, List[int]]]
            Input for this index.
        numpy.ndarray
            Array of one trace.
        """
        assert number_of_samples != 0

        with self.connected():
            self.reset_target()
            for i in tqdm.tqdm(range(number_of_traces)):
                indata = {name: list(func(i)) for name, func in input.items()}
                trace, _ = self.capture_single_trace(
                    number_of_samples,
                    input=b"".join(map(bytes, indata.values())),
                    read_output=False,
                )
                yield (
                    i,
                    indata,
                    trace,
                )

    def capture_single_trace(
        self,
        number_of_samples: int,
        input: Union[Iterable[int], bytes],
        read_output: bool = False,
    ) -> Tuple[numpy.typing.NDArray, Optional[bytes]]:
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
        list(input)
        return np.zeros(number_of_samples), b"" if read_output else None

    @contextlib.contextmanager
    def connected(self, **kwargs):
        """
        Execute code in connected state.

        Returns
        -------
        None

        """
        try:
            self.connect(**kwargs)
            yield
        finally:
            self.disconnect()

    @staticmethod
    def create(platform: str, **kwargs) -> "CaptureDevice":
        """
        Create a CaptureDevice instance based on the given platform.

        Parameters
        ----------
        platform
            The platform used for capturing traces.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        CaptureDevice
            The CaptureDevice object.

        Raises
        ------
        NotImplementedError
            If the platform is invalid or unsupported.

        """
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

    def capture_container(
        self,
        number_of_traces: int,
        input: dict[str, Callable[[int], Union[bytes, List[int]]]],
        additional_fields: dict[
            str,
            Callable[[numpy.typing.NDArray], Union[bytes, List[int]]],
        ],
        number_of_samples=0,
        sample_range: Optional[Tuple[int, int]] = None,
    ) -> lascar.Container:
        """
        Return a CaptureContainer for continuous analysis.

        Parameters
        ----------
        number_of_traces
            The number of traces to capture.
        input
            The dictionary where each key is a string representing the name of the input field,
            and each value is a function that accepts an integer (the index) and returns
            a byte string or list of integers.
            This function is applied to generate the input field data.
        additional_fields
            The dictionary where each key is a string representing the name of an additional field,
            and each value is a function that takes the current trace and returns
            a byte string or list of integers.
            This function is applied to generate the additional field data.
        number_of_samples
            The number of samples to capture, by default 0.
        sample_range
            If not None, only the range of these samples are is used of each trace.

        Returns
        -------
        lascar.Container
            The CaptureContainer.
        """
        return CaptureContainer(
            device=self,
            input=input,
            additional_fields=additional_fields,
            number_of_traces=number_of_traces,
            number_of_samples=number_of_samples,
            sample_range=sample_range,
        )


class CaptureContainer(lascar.AbstractContainer):
    """A CaptureContainer instance for continuous analysis."""

    def __init__(  # noqa: PLR0913
        self,
        device: CaptureDevice,
        input: dict[str, Callable[[int], Union[bytes, List[int]]]],
        additional_fields: dict[
            str,
            Callable[[numpy.typing.NDArray], Union[bytes, List[int]]],
        ],
        number_of_traces: int,
        number_of_samples=0,
        sample_range: Optional[Tuple[int, int]] = None,
        **kargs,
    ):
        """
        Initialize a CaptureContainer instance.

        Parameters
        ----------
        device
            The device to use for capturing.
        input
            The dictionary where each key is a string representing the name of the input field,
            and each value is a function that accepts an integer (the index) and returns
            a byte string or list of integers.
            This function is applied to generate the input field data.
        additional_fields
            The dictionary where each key is a string representing the name of an additional field,
            and each value is a function that takes the current trace and returns
            a byte string or list of integers.
            This function is applied to generate the additional field data.
        number_of_traces
            The number of traces to capture.
        number_of_samples
            The number of samples to capture, by default 0.
        sample_range
            If not None, only the range of these samples are is used of each trace.
        **kargs
            Additional keyword arguments.
        """
        self.device = device
        self.input = input
        self.additional_fields = additional_fields
        self.number_of_samples = number_of_samples
        self.sample_range = sample_range
        lascar.AbstractContainer.__init__(self, number_of_traces, **kargs)

    def generate_trace(self, idx: int) -> lascar.Trace:
        """
        Generate a trace based on a given index.

        Parameters
        ----------
        idx : int
            The index to use for generating the trace.

        Returns
        -------
        lascar.Trace
            The generated trace.
        """
        indata = {name: list(func(idx)) for name, func in self.input.items()}
        with self.device.connected():
            leakage, _ = self.device.capture_single_trace(
                number_of_samples=0,
                input=b"".join(map(bytes, indata.values())),
            )
            self.number_of_samples = len(leakage)
        values_array = input_dict_to_array(
            number_of_traces=1,
            number_of_samples=0,
            input=self.input,
            additional_fields=self.additional_fields,
        )[0]
        for name, value in indata.items():
            values_array[name] = value
        for name, func in self.additional_fields.items():
            values_array[name] = func(values_array)
        return lascar.Trace(
            np.array(
                leakage[self.sample_range[0] : self.sample_range[1]]
                if self.sample_range
                else leakage,
                dtype=np.float32,
            ),
            values_array,
        )

    def generate_trace_batch(self, idx_begin: int, idx_end: int):
        """
        Generate a batch of traces based on a given range of indices.

        Parameters
        ----------
        idx_begin : int
            The beginning index for generating the batch of traces.
        idx_end : int
            The ending index for generating the batch of traces.

        Returns
        -------
        lascar.TraceBatchContainer
            The generated batch of traces.
        """
        traces = self.device.capture(
            number_of_traces=idx_end - idx_begin,
            number_of_samples=self.number_of_samples,
            input=self.input,
            additional_fields=self.additional_fields,
            sample_range=self.sample_range,
        )
        return lascar.TraceBatchContainer(traces["trace"], traces)
