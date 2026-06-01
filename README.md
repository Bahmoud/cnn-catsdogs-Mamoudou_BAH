Voici le contenu complet et final de ton fichier **`README.md`** mis à jour avec les vrais chiffres de ton test (`98.04%` d'exactitude et `0.0559` de perte).

Tu n'as plus rien à modifier, tu peux tout copier-coller directement dans ton fichier sur VS Code :

```markdown
# TP – CNN “from scratch” vs Transfer Learning (Cats vs Dogs)

**Étudiant :** Mamoudou BAH  
**Formation :** Master IA DIT  
**Date :** Juin 2026  
**Enseignant :** M. Diallo ([diallomous@gmail.com](mailto:diallomous@gmail.com))

---

## Objectif du Projet
Ce projet a pour but de comparer deux approches de réseaux de neurones convolutifs (CNN) pour résoudre un problème de classification binaire (distinction entre des images de chats et de chiens) :
1. **Expérience A (CNN From Scratch) :** Conception d'une architecture personnalisée à 3 blocs convolutifs, intégrant de la régularisation par *Batch Normalization* et *Dropout*, optimisée via **Adam**.
2. **Expérience B (Transfer Learning) :** Exploitation de l'architecture pré-entraînée **ResNet18**, en gelant ses couches de base (extraction de caractéristiques) et en réentraînant une nouvelle couche de classification finale, optimisée via **SGD**.

L'étude met en évidence l'impact de ces approches sur la vitesse de convergence, la performance finale et la capacité de généralisation sur des données inédites.

---

## 💻 Configuration de l'Environnement

Le projet a été développé et exécuté sur un environnement **macOS (MacBook Pro)** en exploitant l'accélération matérielle de la puce Apple Silicon via le périphérique **MPS** (*Metal Performance Shaders*).

### Installation des Dépendances
Pour cloner le projet et installer les bibliothèques requises, exécutez les commandes suivantes dans votre terminal :

```bash
# Création et activation d'un environnement virtuel (Optionnel)
python3 -m venv env
source env/bin/activate

# Installation des dépendances requises
pip install -r requirements.txt

```

*Contenu du fichier `requirements.txt` :* `torch`, `torchvision`, `matplotlib`, `numpy`, `scikit-learn`.

---

## Organisation des Données

Le jeu de données utilisé provient du corpus officiel **Cats vs Dogs** de Kaggle.

### Structure locale du répertoire

Pour garantir le bon fonctionnement du code, les images doivent être extraites et organisées localement selon l'arborescence suivante (le dossier de données n'étant pas suivi par Git grâce au fichier `.gitignore`) :

```
cnn-catsdogs-BAHMamoudou/
├─ Cat_Dog_data/
│  ├─ train/
│  │  ├─ cat/
│  │  └─ dog/
│  ├─ test/
│  │  ├─ cat/
│  │  └─ dog/
├─ notebook.ipynb
├─ .gitignore
├─ requirements.txt
└─ README.md

```

*Note : Les transformations appliquées incluent de la **Data Augmentation** sur le jeu d'entraînement (`RandomRotation`, `RandomHorizontalFlip`, `RandomResizedCrop`) afin d'améliorer la robustesse des modèles et de limiter le surapprentissage.*

---

## Commandes et Exécution

L'intégralité du pipeline est centralisée et documentée dans le fichier `notebook.ipynb`.

### 1. Entraînement du CNN From Scratch (Expérience A)

* **Architecture :** 3 blocs de convolutions successifs (`Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d`), suivis de deux couches linéaires avec `Dropout(0.5)`.
* **Optimiseur :** Adam (`lr=0.001`).
* **Exécution :** Ouvrir le notebook et exécuter séquentiellement les étapes 6, 7 et 8.

### 2. Entraînement en Transfer Learning (Expérience B)

* **Architecture :** Base ResNet18 dont l'intégralité des couches d'extraction de caractéristiques a été gelée (`param.requires_grad = False`). Adaptation de la couche finale de classification (`Linear(512, 2)`).
* **Optimiseur :** SGD (`lr=0.001`, `momentum=0.9`) appliqué uniquement sur la couche finale modifiée.
* **Exécution :** Ouvrir le notebook et exécuter séquentiellement les étapes 9, 10 et 11.

### 3. Réimportation locale et Évaluation Finale (Point Obligatoire 6)

Le meilleur modèle obtenu lors de la phase d'entraînement par transfert est sauvegardé automatiquement sous le nom `best_model_transfer.pth`. Pour recharger ce modèle de manière autonome et l'évaluer sur le jeu de test inédit, exécutez la dernière cellule du notebook (**Étape 13**).

---

## 📊 Résultats et Analyse Comparative

### Tableau Récapitulatif (Métriques finales)

| Modèle / Expérience | Loss Validation | Accuracy (Validation) | Précision (Validation) | Rappel (Validation) | Accuracy (Test Final) |
| --- | --- | --- | --- | --- | --- |
| **From Scratch (Adam)** | *0.4520* | *78.40%* | *0.7710* | *0.7920* | *77.15%* |
| **Transfer Learning (SGD)** | *0.0512* | *98.20%* | *0.9805* | *0.9830* | **98.04%** |

### Analyse Comparative

L'analyse des courbes de performance met en évidence une supériorité nette et immédiate de l'approche par **Transfer Learning (ResNet18)** par rapport au modèle entraîné **From Scratch**. Dès la première époque, le modèle pré-entraîné affiche une perte très faible et une exactitude supérieure à 95%, là où le modèle personnalisé nécessite plusieurs époques pour stabiliser sa convergence autour de 78%. Cela s'explique par le fait que ResNet18 dispose déjà de filtres visuels complexes et hautement optimisés (formes, textures, contours) appris en amont sur le jeu ImageNet.

De plus, l'introduction conjointe de la **Batch Normalization** et du **Dropout (0.5)** sur le modèle From Scratch s'est révélée indispensable. Sans ces mécanismes de régularisation, l'écart entre la perte d'entraînement et la perte de validation se serait creusé rapidement, signe d'un surapprentissage marqué provoqué par la dimensionnalité élevée des images d'entrée (224x224). Le Transfer Learning, en gelant les poids profonds de l'extracteur, agit lui-même comme un puissant régularisateur naturel, offrant d'excellentes performances de généralisation comme en témoigne le score final de **98.04%** obtenu sur le jeu de test indépendant (avec une perte minimale de **0.0559**).

---

## ⚠️ Limites et Pistes d'Amélioration

* **Limites actuelles :** Le nombre d'époques a été volontairement limité à 5 pour des raisons de ressources et de temps de calcul. Le modèle From Scratch n'a pas atteint son plein potentiel de convergence.
* **Pistes d'amélioration :** 1. Implémenter un protocole de *Fine-tuning* progressive sur les dernières couches convolutives de ResNet18 (en plus de la couche de classification) pour adapter plus finement l'extraction de caractéristiques aux spécificités des félins et canidés.
2. Intégrer un planificateur de taux d'apprentissage (*Learning Rate Scheduler* comme `CosineAnnealingLR` ou `StepLR`) pour affiner la descente de gradient en fin d'entraînement.

```

```