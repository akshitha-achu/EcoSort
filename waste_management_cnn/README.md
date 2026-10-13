# EcoSort — Six-Class Waste Classification Website

## Files
- `app.py`: Streamlit website
- `predict.py`: command-line predictor for six categories
- `requirements.txt`: dependencies
- Your existing `train_model.py`, `waste_classifier.keras`, and `class_names.txt` are needed.

## Expected folder structure
Keep these files together:
```text
waste_management_cnn/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── waste_classifier.keras
├── class_names.txt
└── dataset/
    ├── e-waste/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── wet_waste/
```

## Run in VS Code (Windows PowerShell)
Activate your existing virtual environment first, then install Streamlit:
```powershell
python -m pip install streamlit
```

Start the website:
```powershell
python -m streamlit run app.py
```
Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Notes
- The app expects the six-class `waste_classifier.keras` and `class_names.txt` created by your current training script.
- This is a prototype. Your reported validation accuracy was about 52%, so predictions may be wrong. Consider improving the model with transfer learning and testing on independent images before presenting it as reliable.
- Model confidence is not a guarantee of correctness.
