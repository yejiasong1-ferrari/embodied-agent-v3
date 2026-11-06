#!/bin/zsh
cd "${0:A:h}"
exec .venv/bin/python run_studio.py --config config/v3.json
