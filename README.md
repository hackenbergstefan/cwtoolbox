# ChipWhisperer Toolbox

Unifying different ChipWhisperer devices and side-channel simulations to a common and easy-to-use API.

## Usage

### Unix

1. Install prerequisites:
   - Git
   - Python >=3.8
   - Make
   - arm-none-eabi-gcc, libnewlib-arm-none-eabi (for Arm victims)
   - avr-gcc, avr-libc (for CWLITEXMEGA)
2. Clone and setup ChipWhisperer:

   ```sh
   git clone https://github.com/newaetech/chipwhisperer.git $HOME/work/chipwhisperer
   export CWFIRMWARE=$HOME/chipwhisperer/hardware/victims/firmware
   ```

   Tip: If you are using VSCode you can omit setting the environment variable globally by adding a `.env` file in you workspace-root with the content `CWFIRMWARE=$HOME/chipwhisperer/hardware/victims/firmware`.

3. If necessary: Adjust udev rules as described here: [https://chipwhisperer.readthedocs.io/en/latest/linux-install.html#installing-chipwhisperer](https://chipwhisperer.readthedocs.io/en/latest/linux-install.html#installing-chipwhisperer)

4. Add `cwtoolbox` as requirement to your project:

   `pyproject.toml`:

   ```toml
   [tool.poetry.dependencies]
   cwtoolbox = {git = "https://github.com/hackenbergstefan/cwtoolbox.git", tag="v0.2.0"}
   ```

   `requirements.txt`:

   ```txt
   git+https://github.com/hackenbergstefan/cwtoolbox.git@v0.2.0#egg=cwtoolbox
   ```

### Windows

1. Install ChipWhisperer as described here: [https://chipwhisperer.readthedocs.io/en/latest/windows-install.html#windows-bundled-installer](https://chipwhisperer.readthedocs.io/en/latest/windows-install.html#windows-bundled-installer).

2. Assuming you installed ChipWhisperer to `C:\cw`.

   Add the following folders to your `PATH`:

   ```txt
   C:\cw\cw\usr\bin;C:\cw\cw\home\portable\armgcc\bin;C:\cw\cw\home\portable\avrgcc\bin
   ```

   Create the following environment variable:

   ```txt
   CWFIRMWAREPATH=C:\cw\cw\home\portable\chipwhisperer\hardware\victims\firmware
   ```

   Tip: If you are using VSCode you can achieve that settings per workspace by adding a `.env` file with the following content:

   ```txt
   PATH=C:\cw\cw\usr\bin;C:\cw\cw\home\portable\armgcc\bin;C:\cw\cw\home\portable\avrgcc\bin;$env["PATH"]
   CWFIRMWAREPATH=C:\cw\cw\home\portable\chipwhisperer\hardware\victims\firmware
   ```

3. Add `cwtoolbox` as requirement to your project:

   `pyproject.toml`:

   ```toml
   [tool.poetry.dependencies]
   cwtoolbox = {git = "https://github.com/hackenbergstefan/cwtoolbox.git", tag="v0.2.0"}
   ```

   `requirements.txt`:

   ```txt
   git+https://github.com/hackenbergstefan/cwtoolbox.git@v0.2.0#egg=cwtoolbox
   ```
