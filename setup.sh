#!/usr/bin/env bash

set -eu
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked
