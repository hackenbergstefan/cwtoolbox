# SPDX-FileCopyrightText: 2023 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

# Target file name (without extension).
# This is the name of the .c-file and of the compiled .hex file.
ifeq ($(TARGET),)
$(error "TARGET is not defined.")
endif

# Path to ChipWhisperer firmware folder
ifeq ($(CWFIRMWAREPATH),)
$(error "CWFIRMWAREPATH is not defined.")
endif

# List C source files here.
# Header files (.h) are automatically pulled in.
SRC += $(TARGET).c

# Pin version of SimpleSerial protocol to 2.1
SS_VER = SS_VER_2_1

CRYPTO_TARGET = NONE

include $(CWFIRMWAREPATH)/simpleserial/Makefile.simpleserial
include $(CWFIRMWAREPATH)/Makefile.inc

CFLAGS += $(ADDITIONAL_CFLAGS)
LDFLAGS += $(ADDITIONAL_LDFLAGS)
