# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

import subprocess

import numpy as np
import pytest

import cwtoolbox


def test_dummy():
    capture = cwtoolbox.CaptureDevice()

    assert (
        capture.capture(number_of_samples=1, number_of_traces=1, input=lambda _: [0])
        == np.zeros(1, dtype=[("trace", "f8", (1,)), ("input", "u1", (1,))])
    ).all()

    assert (
        capture.capture(number_of_samples=10, number_of_traces=42, input=lambda _: [0])
        == np.zeros(42, dtype=[("trace", "f8", (10,)), ("input", "u1", (1,))])
    ).all()


@pytest.mark.chipwhisperer
def test_compile_cw(capture):
    capture.compile(code='asm("nop");')


@pytest.mark.chipwhisperer
def test_compile_cw_additional_flags(capture):
    with pytest.raises(subprocess.CalledProcessError) as exc:
        capture.compile(code='asm("nop");', cflags="-fabc")
        assert "-fabc" in (exc.value)
    with pytest.raises(subprocess.CalledProcessError) as exc:
        capture.compile(code='asm("nop");', ldflags="-fabc")
        assert "-fabc" in (exc.value)


@pytest.mark.chipwhisperer
def test_flash_cw(capture):
    capture.compile(code='asm("nop");')
    capture.flash()


@pytest.mark.chipwhisperer
def test_capture_cw(capture):
    capture.compile(code='asm("nop");')
    capture.flash()
    with capture.connected():
        trace, _ = capture.capture_single_trace(number_of_samples=100, input=[0])
    assert len(trace) == 100

    trace = capture.capture(
        number_of_traces=10,
        number_of_samples=100,
        input=lambda _: [0],
    )
    assert len(trace) == 10
    assert trace.dtype == np.dtype([("trace", "f8", (100,)), ("input", "u1", (1,))])

    if "CWNANO" in capture.cw_platform:
        # CWNANO does not support automatic recognition of number_of_samples
        return

    trace = capture.capture(
        number_of_traces=10,
        input=lambda _: [0],
    )
    assert len(trace) > 0
