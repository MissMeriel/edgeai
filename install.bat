mkdir -p datasets/
tar -xf VisDrone2019-DET-train.zip
python3 -m venv .venv-eai
.venv-eai/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
