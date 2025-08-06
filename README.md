
# 🧠 Machine Learning Projects

Welcome to my collection of machine learning projects! This repository contains explorations, experiments, and implementations of neural networks and deep learning techniques using PyTorch and from-scratch frameworks. Each subfolder represents a specific theme or framework.

---

## 🔸 `makemore/`
This folder contains implementations based on the [makemore](https://github.com/karpathy/makemore) project by Andrej Karpathy, focusing on character-level language models.

### Contents:
- `MLP_nn.ipynb` — A Multi-Layer Perceptron implementation for a simple task.
- `bigram_nn.ipynb` — A character-level bigram language model.
- `names.txt` — A dataset of names used for training the language model.

---

## 🔸 `micro_gpt/`
A work-in-progress folder focusing on digit classification and a scaled-down GPT-like architecture.

### Contents:
- `DigitClassifier_pytorch.ipynb` — A PyTorch model for classifying handwritten digits (MNIST).
- `ImageClassifier.ipynb` — General image classification pipeline using PyTorch.
- `microgpt_inprogress.ipynb` — A minimal GPT model under development.
- `mnist_scaled.npz` — A NumPy-compressed version of the MNIST dataset, scaled for efficient training/testing.
- `playgrnd.ipynb` — A sandbox notebook for testing and prototyping models and ideas.

---

## 🔸 `micrograd_pg/`
This section implements and extends Karpathy's [micrograd](https://github.com/karpathy/micrograd) — a tiny autograd engine.

### Contents:
- `Micrograd.py` — A basic implementation of reverse-mode autodiff.
- `micrograd.ipynb` — Interactive walkthrough of the autodiff engine.
- `micrograd_engine.ipynb` — Enhanced or modular version of the micrograd engine.
- `__pycache__/` — Compiled Python bytecode (auto-generated).

---

## 🛠 Technologies Used
- Python
- PyTorch
- NumPy
- Jupyter Notebooks

---

## 📌 Goals
- Deepen understanding of neural networks from first principles.
- Recreate and extend minimal implementations of well-known models.
- Use PyTorch for real-world and scalable deep learning tasks.

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Install requirements (if any):
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run notebooks with:
   ```bash
   jupyter notebook
   ```

---

## 🧩 TODO
- Finish `microgpt_inprogress.ipynb`.
- Add training/validation metrics visualization.
- Package some modules for reuse.
- Add README for each subproject.

---

## 📬 Contact
For questions or collaboration, feel free to open an issue or contact me directly.
