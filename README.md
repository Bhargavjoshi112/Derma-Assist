# Skin Condition Classifier (learning project)

1. `pip install -r requirements.txt`
2. Put your dataset in a folder with ONE SUB-FOLDER PER CLASS (e.g. data/Eczema/*.jpg).
3. Quick test:  `python train.py --data data --max-per-class 60 --epochs 2`
4. Real run:    `python train.py --data data --max-per-class 1500 --epochs 12`
5. Web app:     `python app.py`  ->  http://127.0.0.1:5001

Training writes to `models/`: `skin_model.pt` (the model), `report.txt` (test results),
`metrics.json`, `splits.json` (which image went to train/val/test).

Educational only - not a medical device.
