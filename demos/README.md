# Demo Notebooks

Interactive examples demonstrating the AUC optimization framework.

## Setup

1. **Install dependencies** (from project root):
```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
```

2. **Launch Jupyter**:
```bash
   jupyter notebook
```

3. **Open a notebook** and run all cells!

## Available Demos

### `svm_example/svm_example.ipynb`
Demonstrates ALM optimization on synthetic datasets with baseline comparisons.

- **Dataset**: Synthetic 2D and 25D binary classification
- **Methods**: Prox-SGD, PyTorch BCE, LibAUC
- **Runtime**: ~2-5 minutes

## Troubleshooting

**"No module named 'aucopt'"**
- Notebooks handle this automatically via path setup in first cell
- Or install: `pip install -e .` from project root

**"ModuleNotFoundError: libauc"**
- Optional baseline: `pip install libauc`
- Or skip LibAUC comparison cells

**Notebook kernel crashes**
- Reduce dataset size: change `m=2000` to `m=500`