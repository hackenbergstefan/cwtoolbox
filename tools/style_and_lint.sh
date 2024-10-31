#!/bin/bash

# SPDX-FileCopyrightText: 2024 Stefan Hackenberg
#
# SPDX-License-Identifier: MIT

set -xeo pipefail
shopt -s globstar

pushd "$(dirname "$0")/.."

CORE_SRC="cwtoolbox/"
ALL_SRC="cwtoolbox/ tests/"

ruff format $ALL_SRC
ruff check --fix --exit-zero $CORE_SRC
git diff --quiet || (git diff > changes.patch)
git diff --color=always --exit-code
ruff check --silent $CORE_SRC
mypy $CORE_SRC

popd
