import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# CONFIG
ML_MODEL_PATH = "../models/mushroom_machine_learning.pkl"
DL_MODEL_PATH = "../models/mushroom_deep_learning.keras"
IMG_SIZE = (128, 128)

# Classes pour le DL 
CLASS_NAMES_DL = ["Amanita muscaria", "Coprinus comatus", "Laetiporus sulphureus"]

# Features pour le ML
FEATURE_NAMES = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
    'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat'
]

# Options pour chaque feature
FEATURE_OPTIONS = {
    'cap-shape': ['b', 'c', 'x', 'f', 'k', 's'],
    'cap-surface': ['f', 'g', 'y', 's'],
    'cap-color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    'bruises': ['t', 'f'],
    'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    'gill-attachment': ['a', 'd', 'f', 'n'],
    'gill-spacing': ['c', 'w', 'd'],
    'gill-size': ['b', 'n'],
    'gill-color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    'stalk-shape': ['e', 't'],
    'stalk-surface-above-ring': ['f', 'y', 'k', 's'],
    'stalk-surface-below-ring': ['f', 'y', 'k', 's'],
    'stalk-color-above-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    'stalk-color-below-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    'veil-type': ['p', 'u'],
    'veil-color': ['n', 'o', 'w', 'y'],
    'ring-number': ['n', 'o', 't'],
    'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
    'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
    'population': ['a', 'c', 'n', 's', 'v', 'y'],
    'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd']
}

# MODELS
@st.cache_resource
def load_ml_pipeline(path):
    return joblib.load(path)

@st.cache_resource
def load_dl_model(path):
    return load_model(path)

# STREAMLIT APP
st.title("🍄 Classification de Champignons")

st.markdown("""
Cette application permet de classifier des champignons avec deux approches :
1. **Machine Learning** : à partir de caractéristiques manuelles
2. **Deep Learning** : à partir d'une photo
""")

# Tabs pour séparer ML et DL
tab1, tab2 = st.tabs(["🌲 Machine Learning", "📷 Deep Learning"])

# TAB 1 : Machine Learning
with tab1:
    st.header("Classification par caractéristiques")
    st.markdown("Renseignez les caractéristiques du champignon :")
    
    # Charger le modèle ML
    try:
        ml_pipeline = load_ml_pipeline(ML_MODEL_PATH)
        ml_loaded = True
    except Exception as e:
        st.error(f"Erreur chargement modèle ML : {e}")
        ml_loaded = False
    
    if ml_loaded:
        # Créer les inputs pour chaque feature
        col1, col2, col3 = st.columns(3)
        input_dict = {}
        
        for i, feature in enumerate(FEATURE_NAMES):
            if feature in FEATURE_OPTIONS:
                if i % 3 == 0:
                    with col1:
                        input_dict[feature] = st.selectbox(feature, FEATURE_OPTIONS[feature], key=feature)
                elif i % 3 == 1:
                    with col2:
                        input_dict[feature] = st.selectbox(feature, FEATURE_OPTIONS[feature], key=feature)
                else:
                    with col3:
                        input_dict[feature] = st.selectbox(feature, FEATURE_OPTIONS[feature], key=feature)
        
        # Bouton de prédiction
        if st.button("🔍 Prédire (ML)", key="btn_ml"):
            input_df = pd.DataFrame([input_dict])
            
            try:
                prediction = ml_pipeline.predict(input_df)[0]
                
                if prediction == 'e':
                    st.success("✅ **Comestible** (edible)")
                else:
                    st.error("☠️ **Vénéneux** (poisonous)")
                    
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

# TAB 2 : Deep Learning
with tab2:
    st.header("Classification par image")
    st.markdown("Téléversez une photo de champignon :")
    
    # Charger le modèle DL
    try:
        dl_model = load_dl_model(DL_MODEL_PATH)
        dl_loaded = True
    except Exception as e:
        st.error(f"Erreur chargement modèle DL : {e}")
        dl_loaded = False
    
    if dl_loaded:
        uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Afficher l'image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Image chargée", width=300)
            
            # Bouton de prédiction
            if st.button("🔍 Prédire (DL)", key="btn_dl"):
                # Preprocessing
                img_resized = img.resize(IMG_SIZE)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prédiction
                predictions = dl_model.predict(img_array)
                pred_idx = np.argmax(predictions[0])
                confidence = predictions[0][pred_idx]
                pred_class = CLASS_NAMES_DL[pred_idx]
                
                # Affichage
                st.markdown("---")
                st.subheader("Résultat")
                st.write(f"**Classe prédite :** {pred_class}")
                st.write(f"**Confiance :** {confidence:.2%}")
                
                # Afficher toutes les probabilités
                st.markdown("**Probabilités par classe :**")
                for i, class_name in enumerate(CLASS_NAMES_DL):
                    st.progress(float(predictions[0][i]), text=f"{class_name}: {predictions[0][i]:.2%}")

# FOOTER
st.markdown("---")
st.markdown("*Projet Alyra - Classification de champignons (ML + DL)*")