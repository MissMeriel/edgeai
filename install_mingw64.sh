#/usr/bin/bash
alias pycall="python"
mkdir -p datasets/
pycall -m venv .venv-eai
source .venv-eai/Scripts/activate
pycall -m pip install --upgrade pip
pycall -m pip install -r requirements.txt