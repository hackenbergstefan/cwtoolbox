# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

import importlib
import importlib.resources
import logging
import os
import subprocess
import tempfile
import time
import warnings
import xmlrpc
import xmlrpc.client
from pathlib import Path
from typing import Iterable, Optional, Type, Union, cast

import jinja2
import numpy as np
import numpy.typing

from .. import CaptureDevice  # pylint: disable=cyclic-import

warnings.filterwarnings(action="ignore", module="pkg_resources.*")

# Import chipwhisperer after filtering warnings
import chipwhisperer as cw  # pylint: disable=wrong-import-order,wrong-import-position


class CwCaptureSimpleSerial(CaptureDevice):
    TARGET_TIMEOUT_RETRY = 100

    def __init__(self, cw_platform: str):
        self.buildfolder = Path(".") / "build"
        self.cw_platform = cw_platform
        self.scope: Optional[cw.scopes.ScopeTypes] = None
        self.target: Optional[cw.targets.TargetTypes] = None
        self.cw_serial_number: Optional[str] = None

    def reset_target(self):
        # pylint: disable=line-too-long
        # From https://github.com/newaetech/chipwhisperer-jupyter/blob/8cbf8f56f27c696e4d636a54ad13b7119570d583/Setup_Scripts/Setup_Generic.ipynb
        if self.cw_platform in ("CW303", "CWLITEXMEGA"):
            self.scope.io.pdic = "low"
            time.sleep(0.1)
            self.scope.io.pdic = "high_z"  # XMEGA doesn't like pdic driven high
            time.sleep(0.1)  # xmega needs more startup time
        elif "neorv32" in self.cw_platform.lower():
            raise IOError(
                "Default iCE40 neorv32 build does not have external reset - reprogram device to reset"
            )
        elif self.cw_platform == "CW308_SAM4S":
            self.scope.io.nrst = "low"
            time.sleep(0.25)
            self.scope.io.nrst = "high_z"
            time.sleep(0.25)
        else:
            self.scope.io.nrst = "low"
            time.sleep(0.05)
            self.scope.io.nrst = "high_z"
            time.sleep(0.05)

    def capture_single_trace(
        self,
        number_of_samples: int,
        input: Iterable[int],
    ) -> numpy.typing.NDArray:
        assert self.scope is not None
        assert self.target is not None
        self.scope.adc.samples = number_of_samples
        self.scope.arm()
        self.target.flush()
        self.target.simpleserial_write(0x01, bytes(input))

        ret = self.scope.capture()

        for i in range(self.TARGET_TIMEOUT_RETRY):
            if self.target.is_done():
                break
            time.sleep(0.05)
            if i == self.TARGET_TIMEOUT_RETRY - 1:
                raise TimeoutError("Target did not finish operation")
        if ret:
            raise TimeoutError("Timeout happened during capture")

        return self.scope.get_last_trace()

    def compile(
        self,
        file: Optional[Union[str, Path]] = None,
        code: Optional[str] = None,
        cflags: str = "",
        ldflags: str = "",
    ):
        # Render code
        self.buildfolder.mkdir(exist_ok=True)
        self._render_code(file=file, code=code)

        # Extract makefile
        (self.buildfolder / "simpleserial.mak").write_text(
            importlib.resources.read_text("cwtoolbox.cw", "simpleserial.mak")
        )

        # Compile
        cmd = [
            "make",
            "-f",
            f"{(self.buildfolder / 'simpleserial.mak').absolute()}",
            f"PLATFORM={self.cw_platform}",
            f"FIRMWAREPATH={os.environ['CWFIRMWAREPATH']}",
            f"TARGET={(self.buildfolder / 'generic_simpleserial.c').stem}",
            f"ADDITIONAL_CFLAGS={cflags}",
            f"ADDITIONAL_LDFLAGS={ldflags}",
            "clean",
            "allquick",
        ]
        logging.getLogger(__name__).debug(
            f"Executing '{' '.join(cmd)}' in '{self.buildfolder.absolute()}'"
        )
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.buildfolder.absolute(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            logging.getLogger(__name__).debug(
                f"\x1b[32m✓\x1b[0m {proc.stdout.decode()}"
            )
        except subprocess.CalledProcessError as e:
            logging.getLogger(__name__).fatal(
                f'\x1b[31m✗ "{" ".join(e.args[1])}" returned:\x1b[0m\n {e.stderr.decode()}'
            )
            raise

    def _render_code(
        self,
        file: Optional[Union[str, Path]] = None,
        code: Optional[str] = None,
    ):
        template = jinja2.Environment().from_string(
            importlib.resources.read_text("cwtoolbox.cw", "generic_simpleserial.c.j2")
        )
        rendered = template.render(code=code, fromfile=file)
        (self.buildfolder / "generic_simpleserial.c").write_text(rendered)

    def flash(self, file: Optional[str] = None):
        with self.connected():
            if self.cw_platform == "CWLITEXMEGA":
                prog: Type[cw.programmers.Programmer] = cw.programmers.XMEGAProgrammer
            elif (
                self.cw_platform == "CWLITEARM"
                or self.cw_platform == "CWNANO"
                or "STM" in self.cw_platform
            ):
                prog = cw.programmers.STM32FProgrammer
            assert self.scope is not None
            cw.program_target(
                scope=self.scope,
                prog_type=prog,
                fw_path=str(
                    file
                    or self.buildfolder / f"generic_simpleserial-{self.cw_platform}.hex"
                ),
            )

    def _set_cw_serial_number(self):
        if self.cw_serial_number:
            return

        # NOTE: CWLITEARM and CWLITEXMEGA refer to the same device name.
        cw_names = {
            "CWNANO": "ChipWhisperer-Nano",
            "CWLITE": "ChipWhisperer-Lite",
        }
        platform_short = self.cw_platform[:6]
        for dev in cw.list_devices():
            if dev["name"] == cw_names[platform_short]:
                # Distinguish between cwlitearm and cwlitexmega
                if self.cw_platform == "CWLITEXMEGA" and not self._victim_is_xmega(
                    dev["sn"]
                ):
                    continue
                self.cw_serial_number = dev["sn"]
                return
        raise OSError(f"No ChipWhisperer for '{self.cw_platform}' found.")

    def _victim_is_xmega(self, serial_number: str):
        # pylint: disable=line-too-long
        # From: https://github.com/newaetech/chipwhisperer/blob/38e2ddca5bce4e862440af7de4c83486c54a614d/software/chipwhisperer/__init__.py#L153
        scope = cw.scope(sn=serial_number)
        scope.default_setup()
        prog = cw.programmers.XMEGAProgrammer()
        prog.scope = scope
        try:
            prog.open()
            prog.find()
            return True
        except:  # pylint: disable=bare-except
            return False
        finally:
            if isinstance(scope, cw.scopes.OpenADC):
                scope.io.pdic = 0
                time.sleep(0.05)
                scope.io.pdic = None
                time.sleep(0.05)
            prog.close()
            scope.dis()

    def connect(self):
        self._set_cw_serial_number()
        self.scope = cw.scope(sn=self.cw_serial_number)
        self.scope.default_setup()
        self.target = cw.target(self.scope, target_type=cw.targets.SimpleSerial2)

    def disconnect(self):
        if self.scope is not None:
            self.scope.dis()
            self.target.dis()
            self.scope = None
            self.target = None
            self.cw_serial_number = None


class CwCaptureSimpleSerialRpcClient(CwCaptureSimpleSerial):
    def __init__(self, cw_platform: str, serveruri: str):
        super().__init__(cw_platform)
        self.proxy = xmlrpc.client.ServerProxy(
            serveruri,
            allow_none=True,
            use_builtin_types=True,
        )
        self.proxy.set_cw_platform(self.cw_platform)

    def flash(self, file: Optional[str] = None):
        self.proxy.flash(
            (
                Path(file)
                if file
                else (self.buildfolder / f"generic_simpleserial-{self.cw_platform}.hex")
            ).read_bytes()
        )

    def capture_single_trace(
        self,
        number_of_samples: int,
        input: Iterable[int],
    ) -> numpy.typing.NDArray:
        data = cast(
            bytes,
            self.proxy.capture_single_trace(
                number_of_samples,
                np.array(input).tobytes(),
            ),
        )
        return np.frombuffer(data)

    def reset_target(self):
        self.proxy.reset_target()

    def connect(self):
        self.proxy.connect()

    def disconnect(self):
        self.proxy.disconnect()


class CwCaptureSimpleSerialRpcClientResourceServer(CwCaptureSimpleSerialRpcClient):
    def __init__(self, cw_platform: str, resource_server_uri: str):
        self.resource_server_uri = resource_server_uri
        self.resource_ticket: Optional[str] = None
        self.resource_properties: Optional[dict] = None
        self.cw_platform = cw_platform
        self.acquire()
        assert self.resource_properties is not None
        super().__init__(
            cw_platform,
            serveruri=self.resource_properties["xmlrpc-server-uri"],
        )
        self.cw_serial_number = self.resource_properties["cwserial"]

    def acquire(self):
        self.release()
        resource_server = xmlrpc.client.ServerProxy(self.resource_server_uri)
        self.resource_ticket, self.resource_properties = resource_server.acquire(
            "ChipWhisperer",
            {"cwplatform": self.cw_platform},
            0.1,
        )
        if not self.resource_ticket:
            raise RuntimeError("Requested resource could not be acquired.")
        logging.getLogger(__name__).debug(
            "CwCaptureSimpleSerialRpcClientResourceServer:acquire: "
            f"{self.resource_ticket}, {self.resource_properties}"
        )

    def release(self):
        if self.resource_ticket is None:
            return

        resource_server = xmlrpc.client.ServerProxy(self.resource_server_uri)
        resource_server.release(self.resource_ticket)
        self.resource_ticket = None


class CwCaptureSimpleSerialRpcService(CwCaptureSimpleSerial):
    def __init__(self):
        super().__init__(cw_platform=None)

    def get_cw_platform(self):
        return self.cw_platform

    def set_cw_platform(self, cw_platform):
        self.cw_platform = cw_platform

    def flash(  # type: ignore[override] # pylint: disable=arguments-renamed
        self,
        filedata: bytes,
    ):
        with tempfile.NamedTemporaryFile(suffix=".hex") as file:
            file.write(filedata)
            super().flash(file=file.name)

    def capture_single_trace(  # type: ignore[override]
        self,
        number_of_samples: int,
        input: bytes,
    ):
        input = np.frombuffer(input, dtype="uint8")
        trace = super().capture_single_trace(number_of_samples, input)
        return trace.tobytes()
