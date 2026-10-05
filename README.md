# Graduate Admission Predictor — ANN (Multi-Layer Perceptron)

**Soft Computing Mini Project**
**Name:** Aakansha Kasana  **ID:** 2026246809

## Problem
Predict whether a student is **admitted (1)** or **rejected (0)** using:
- GRE score (260–340)
- CGPA (6–10)
- Research experience (0 / 1)

## Model
Fully connected ANN built from scratch with NumPy:

`3 inputs → 6 hidden (ReLU) → 4 hidden (ReLU) → 1 output (Sigmoid)`

- **Loss:** Binary Cross-Entropy
- **Learning:** Backpropagation + gradient descent (lr = 0.1, 1500 epochs)
- **Data:** synthetic, reproducible (seed 42), 80/20 train-test split, z-score normalised

## Results
- Test accuracy: **79%**
- Output: `training_curves.png` (loss/accuracy), `decision_map.png` (decision boundary on GRE vs CGPA)

## Run
```bash
pip install numpy matplotlib
python admission_predictor_ann.py
```

## Files
- `admission_predictor_ann.py` — dataset, network, training, evaluation, plots
- `training_curves.png`, `decision_map.png` — outputs
