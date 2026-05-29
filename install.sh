#/usr/bin/bash
alias pycall="python"
mkdir -p datasets/
pycall -m venv .venv-eai
source .venv-eai/bin/activate
pycall -m pip install --upgrade pip
pycall -m pip install -r requirements.txt
