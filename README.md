# 🚀 LearningAI - My AI/ML Learning Journey

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)](#)

A comprehensive repository documenting my journey through AI and Machine Learning fundamentals, from basic neural networks to physics-based models.

[✨ Features](#features) • [📁 Directory Structure](#-directory-structure) • [🎓 Learning Path](#-learning-path) • [🔗 Resources](#-resources)

</div>

---

## 🎯 Overview

This repository serves as my personal learning hub for AI/ML concepts and implementations. It contains:
- **Foundational concepts** from tutorials and textbooks
- **Hands-on implementations** using PyTorch
- **Physics-based models** and simulations
- **Progressive complexity** from basics to advanced topics

Each project includes detailed notebooks with explanations, code, and results.

---

## ✨ Features

- 📚 **Tutorial-Based Learning** - Follow along with industry-standard courses
- 🔬 **Practical Implementations** - Real code, not just theory
- 📊 **Jupyter Notebooks** - Interactive learning with visualizations
- 🎓 **Well-Documented** - Comments and explanations throughout
- 🧠 **Progressive Difficulty** - Start simple, build complexity
- 🚀 **Production-Ready Code** - Clean, modular implementations

---

## 📁 Directory Structure

```
LearningAI/
├── Basics/
│   ├── biagram_model/
│   │   ├── makemore.ipynb              # 📌 [Andrej Karpathy - Makemore Tutorial]
│   │   └── names.txt                   # Dataset: 32K English baby names
│   └── py_torch_basics.py              # PyTorch fundamentals
│
├── PhysicsBased/
│   └── basics.py                       # Physics-based model implementations
│
└── README.md                           # You are here! 👈
```

### 📚 Basics Directory
The `Basics` folder contains foundational concepts and implementations:

#### 🎬 Biagram Model (Makemore)
- **Tutorial**: [Makemore - Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo)
- **Content**: Building a character-level language model to generate baby names
- **Key Concepts**:
  - Bigram probability distributions
  - Character encoding/decoding
  - PyTorch tensor operations
  - Probability sampling with generators
- **Dataset**: 32,033 English baby names
- **Output**: Generated synthetic names based on learned patterns

#### 🔧 PyTorch Basics
- Fundamental PyTorch operations
- Tensor manipulations
- Basic neural network concepts

### 🧬 PhysicsBased Directory
Advanced implementations incorporating physics principles:
- Physics-informed neural networks
- Conservation laws
- Differential equations
- Simulation-based learning

---

## 🎓 Learning Path

### Phase 1: Foundations (Current) ✅
- [x] Character-level language models
- [x] Probability distributions
- [x] PyTorch basics
- [ ] Multi-layer perceptrons

### Phase 2: Intermediate (Upcoming)
- [ ] Recurrent Neural Networks (RNNs)
- [ ] Attention mechanisms
- [ ] Transformer architectures
- [ ] Fine-tuning pre-trained models

### Phase 3: Advanced (Future)
- [ ] Physics-informed neural networks (PINNs)
- [ ] Graph neural networks
- [ ] Reinforcement learning
- [ ] Diffusion models

---

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.12
uv (Fast Python package installer)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/LearningAI.git
cd LearningAI

# Create and sync virtual environment with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Running the Notebooks
```bash
# Start Jupyter with uv
uv run jupyter notebook

# Open and run:
# - Basics/biagram_model/makemore.ipynb
# - Other notebooks as you explore
```

### Running Python Scripts
```bash
# Run PyTorch basics
uv run python Basics/py_torch_basics.py

# Run physics-based models
uv run python PhysicsBased/basics.py
```

### Installing Additional Dependencies
```bash
# Add new packages to the project
uv pip install package_name

# Or use uv add for managed dependencies
uv add package_name
```

---

## 📊 Project Highlights

### Makemore - Character-Level Language Model
This project implements a simple but elegant language model:

```
📈 Model Architecture:
├── Input Layer: Character embeddings
├── Bigram Statistics: Probability distributions
└── Output Layer: Next character prediction

📊 Results:
├── Vocabulary Size: 27 characters (a-z + special token)
├── Training Data: 32,033 names
└── Sample Output: Generated realistic names from learned patterns
```

**Key Learnings:**
- Building probability distributions from data
- Character encoding strategies
- Sampling from distributions
- Data visualization with matplotlib

---

## 🔗 Resources

### Recommended Tutorials & Courses
- **[Makemore Series](https://www.youtube.com/watch?v=PaCmpygFfXo)** - Andrej Karpathy's character-level language models
- **[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrZiCoo0dEwnNB1zrVzay6fHMzxqn)** - Complete ML foundation course
- **[PyTorch Official Tutorials](https://pytorch.org/tutorials/)** - Learn PyTorch from the source

### Books
- "Deep Learning" by Goodfellow, Bengio, and Courville
- "Neural Networks from Scratch" by Trask
- "Physics-Informed Machine Learning" - Recent research papers

### Tools & Libraries
- 🔥 **PyTorch** - Deep learning framework
- 📓 **Jupyter Notebooks** - Interactive computing
- 📊 **Matplotlib** - Data visualization
- 🔢 **NumPy** - Numerical computing

---

## 💡 Key Concepts Covered

| Concept | Status | Location |
|---------|--------|----------|
| Character Encoding | ✅ Complete | `Basics/biagram_model/` |
| Probability Distributions | ✅ Complete | `Basics/biagram_model/` |
| PyTorch Tensors | ✅ Complete | `Basics/py_torch_basics.py` |
| Physics-Based Models | 🔄 In Progress | `PhysicsBased/` |
| RNNs | ⏳ Planned | TBD |
| Transformers | ⏳ Planned | TBD |

---

## 🤝 Contributing

This is a personal learning repository, but I welcome suggestions! Feel free to:
- Report issues or corrections
- Suggest improvements
- Share learning resources
- Discuss concepts

---

## 📝 Notes & Documentation

Each file includes:
- **Comments**: Inline explanations of complex logic
- **Docstrings**: Function and module documentation
- **Markdown cells** (in notebooks): Concept explanations
- **Output examples**: Expected results and visualizations

---

## 🎯 Goals & Objectives

**Short Term (Next 3 months):**
- ✅ Master character-level language models
- ⏳ Implement RNNs from scratch
- ⏳ Build a simple transformer

**Medium Term (Next 6 months):**
- ⏳ Implement attention mechanisms
- ⏳ Explore transfer learning
- ⏳ Create physics-informed models

**Long Term (Next Year):**
- ⏳ Build advanced neural architectures
- ⏳ Contribute to open-source ML projects
- ⏳ Create production-ready models

---

## 🔮 Future Additions

- [ ] Recurrent Neural Networks (RNNs)
- [ ] Long Short-Term Memory (LSTM) networks
- [ ] Gated Recurrent Units (GRUs)
- [ ] Attention Mechanisms
- [ ] Transformer from scratch
- [ ] Vision Transformers (ViT)
- [ ] Physics-Informed Neural Networks (PINNs)
- [ ] Graph Neural Networks (GNNs)
- [ ] Reinforcement Learning fundamentals
- [ ] Generative models (VAE, GAN, Diffusion)

---

## 📚 Notebook Descriptions

### `Basics/biagram_model/makemore.ipynb`
**Status:** ✅ Complete  
**Time to Complete:** ~2 hours  
**Difficulty:** Beginner  

A comprehensive walkthrough of building a character-level language model using bigram statistics. Starting from raw data loading to generating synthetic names, this notebook covers all the fundamentals needed to understand how language models work at the most basic level.

**Topics Covered:**
- Data loading and preprocessing
- Bigram extraction and counting
- Probability matrix construction
- Visualization of statistics
- Sampling from distributions
- Name generation

---

## 🛠️ Tech Stack

```
Backend:
├── Python 3.12+
├── PyTorch 2.0+
├── NumPy
└── Matplotlib

Development:
├── Jupyter Notebook
├── Git & GitHub
└── VS Code / Cursor IDE
```

---

## 📞 Get In Touch

- **GitHub**: [Your GitHub Profile]
- **Twitter**: [@YourHandle]
- **LinkedIn**: [Your LinkedIn]

---

## ⭐ If This Repo Helped You!

If you found this repository useful for your own learning journey, please consider:
- ⭐ Starring this repository
- 🔄 Sharing it with others
- 💬 Leaving feedback and suggestions

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

### 🚀 Happy Learning! Keep Building, Keep Improving!

**Last Updated:** December 22, 2025  
**Last Modified:** 2 weeks ago

[⬆ Back to Top](#-learningai---my-aiml-learning-journey)

</div>

