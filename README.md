# PSDS MAHIR — Convolutional Neural Networks (CNN) Notebook Series 🧠🖼️

[![Jupyter Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=jupyter)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Last Commit](https://img.shields.io/github/last-commit/frenskuy/PSDS-MAHIR-CNN-?style=for-the-badge)](https://github.com/frenskuy/PSDS-MAHIR-CNN-/commits/main)

A **hands-on, beginner-friendly** set of Jupyter notebooks that walk you through **CNN fundamentals** — from **data augmentation** and **convolution/pooling** to **training** and **model evaluation**.  
Designed as a **learning path** you can follow in order (1 → 9).

---

## Table of Contents
- [What’s Inside](#whats-inside)
- [Learning Path](#learning-path)
- [Quick Start](#quick-start)
- [Recommended Environment](#recommended-environment)
- [How to Run](#how-to-run)
- [Tips](#tips)
- [Contributing](#contributing)
- [License](#license)

---

## What’s Inside

This repository contains the following notebooks:

| # | Notebook | Topic | Open in Colab |
|---:|---|---|---|
| 1 | **Augmentation_Data.ipynb** | Image augmentation basics + visualization | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/1.%20Augmentation_Data.ipynb) |
| 2 | **Convolutional_Layer.ipynb** | Convolution layer intuition + operations | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/2.%20Convolutional_Layer.ipynb) |
| 3 | **Pooling_Layer.ipynb** | Max/Avg pooling, downsampling intuition | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/3.%20Pooling_Layer.ipynb) |
| 4 | **Activation_Function.ipynb** | ReLU/Tanh/etc. + non-linearity role | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/4.%20Activation_Function.ipynb) |
| 5 | **Fully_Connected_Layer.ipynb** | FC layer basics for classification | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/5.%20Fully_Connected_Layer.ipynb) |
| 6 | **Fully_Connected_Layer_bias_vs_no_bias.ipynb** | Bias vs no-bias comparison | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/6.%20Fully_Connected_Layer_bias_vs_no_bias.ipynb) |
| 7 | **Training_Model.ipynb** | Training pipeline (train/val loop, loss, acc) | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/7.%20Training_Model.ipynb) |
| 8 | **Training_Model_with_Augmentation.ipynb** | Training with augmentation | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/8.%20Training_Model_with_Augmentation.ipynb) |
| 9 | **Model_Evaluation.ipynb** | Evaluation: confusion matrix, report, metrics | [Open](https://colab.research.google.com/github/frenskuy/PSDS-MAHIR-CNN-/blob/main/9.%20Model_Evaluation.ipynb) |

> 💡 Tip: Follow the notebooks in order for the smoothest learning curve.

---

## Learning Path

If you’re new to CNNs, here’s the recommended flow:

1) **Augment the data** → build intuition about robustness and generalization  
2) **Convolution** → feature extraction fundamentals  
3) **Pooling** → spatial reduction & translation tolerance  
4) **Activation** → why “deep” needs non-linearity  
5) **Fully Connected** → turning features into class scores  
6) **Bias vs No Bias** → small detail, big effect on representation  
7) **Training** → loss, accuracy, backprop loop  
8) **Training + Augmentation** → reduce overfitting  
9) **Evaluation** → metrics that actually matter

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/frenskuy/PSDS-MAHIR-CNN-.git
cd PSDS-MAHIR-CNN-
````

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -U pip
pip install jupyter numpy matplotlib pillow scikit-learn
# Deep Learning stack (choose one)
pip install torch torchvision torchaudio
```

---

## Recommended Environment

* **Python**: 3.9+
* **Jupyter**: Notebook / Lab
* **Framework**: PyTorch (recommended)
* **Optional**: GPU runtime (Colab / CUDA) for faster training

---

## How to Run

### Option A — Run locally

```bash
jupyter notebook
```

Then open any notebook and run cells top-to-bottom.

### Option B — Run in Google Colab

Use the **Open in Colab** links in the table above.

---

## Tips

* **Training feels slow?**

  * Try GPU (Colab: `Runtime → Change runtime type → GPU`)
  * Reduce image resolution or batch size (for small machines)
  * Use fewer epochs while experimenting

* **Getting low accuracy?**

  * Check class imbalance & labels
  * Add augmentation (Notebook 8)
  * Tune learning rate, optimizer, and regularization

---

## Contributing

Contributions are welcome 🙌
If you’d like to improve explanations, add diagrams, or include more experiments:

1. Fork this repo
2. Create a new branch
3. Submit a Pull Request

---

## License

No license file is included yet.
If you plan to reuse this material publicly, please open an issue or a PR to discuss adding a license.

---


If you found this helpful, consider leaving a ⭐ to support the project!

