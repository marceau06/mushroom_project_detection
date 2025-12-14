# 🍄 Mushroom Detection Project

Projet de classification de champignons utilisant le Machine Learning et le Deep Learning.

---

## 📋 Description

Ce projet propose deux approches de classification :

| Partie | Description | Modèle |
|--------|-------------|--------|
| **Machine Learning** | Classification selon la comestibilité (comestible / vénéneux) | Random Forest |
| **Deep Learning** | Classification d'espèces de champignons | CNN |

---

## 🚀 Installation
```bash
# Cloner le repo
git clone https://github.com/marceau06/mushroom_project_detection.git
cd mushroom_project_detection

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Installer les dépendances
pip install -r requirements.txt
```

---

## ▶️ Lancer l'application
```bash
streamlit run app/app.py
```

---

## 📁 Structure du projet
```
mushroom_project_detection/
├── app/
│   └── app.py                 # Application Streamlit
├── dataset/                   # Données d'entraînement
├── models/
│   ├── mushroom_machine_learning.pkl
│   └── mushroom_deep_learning.keras
├── notebooks/
│   ├── mushroom_ml.ipynb      # Notebook Machine Learning
│   └── mushroom_dl.ipynb      # Notebook Deep Learning
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies

- Python
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Pandas / NumPy

---

## 👤 Auteur

**Marceau LÊ** - Projet Alyra
