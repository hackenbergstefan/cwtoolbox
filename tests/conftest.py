# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

from functools import partialmethod

import pytest
import tqdm

import cwtoolbox

# Disable tqdm output for tests
tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)


def pytest_addoption(parser):
    parser.addoption(
        "--cw-platform",
        choices=(
            "CWLITEXMEGA",
            "CWLITEARM",
            "CWNANO",
            "XMLRPCCWLITEXMEGA",
            "XMLRPCCWLITEARM",
            "XMLRPCCWNANO",
        ),
        help="ChipWhisperer test platform.",
        required=True,
    )

    parser.addoption(
        "--resource-server-uri",
        help="URI of Resource Server.",
    )


@pytest.fixture
def capture(request):
    capture_device = None
    try:
        capture_device = cwtoolbox.CaptureDevice.create(
            request.config.option.cw_platform,
            **(
                {"resource_server_uri": request.config.option.resource_server_uri}
                if request.config.option.resource_server_uri
                else {}
            )
        )
        yield capture_device
    finally:
        getattr(capture_device, "release", lambda: None)()


@pytest.fixture
def xmlrpc_server(request):
    return request.config.option.xmlrpc_server
