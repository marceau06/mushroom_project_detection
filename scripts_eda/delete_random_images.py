import os
import random

# Chemin du dataset
dataset_path = "./dataset/train/"

# Classes (dossiers)
classes = ["Amanita muscaria", "Laetiporus sulphureus"]

# Récupérer toutes les images
all_images = []
for cls in classes:
    class_path = os.path.join(dataset_path, cls)
    for img_name in os.listdir(class_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(os.path.join(class_path, img_name))

print(f"Total images trouvées : {len(all_images)}")

# Sélectionner 200 images aléatoirement
images_to_delete = random.sample(all_images, min(200, len(all_images)))

# Supprimer les images
for img_path in images_to_delete:
    os.remove(img_path)

print(f"✅ {len(images_to_delete)} images supprimées")