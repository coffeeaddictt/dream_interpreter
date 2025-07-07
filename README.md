# Synthétiseur de rêves 🌙

**Dream Interpreter** est une application web interactive qui vous permet de :

- **Enregistrer** vos rêves à l’oral.  
- **Transcrire** automatiquement vos enregistrements grâce à l’API Whisper (OpenAI).  
- **Analyser** l’émotion dominante (heureux, stressant, neutre) via GPT-4.  
- **Générer** une illustration onirique avec DALL·E ou Stable Diffusion.  
- **Consulter** un tableau de bord personnel répertoriant tous vos rêves.

---

## 🚀 Fonctionnalités

1. **Record Dream** : lancez l’enregistrement audio directement dans l’app (bouton “Start/Stop”).  
2. **Upload Dream** : téléversez un fichier `.wav` ou `.mp3`.  
3. **Analyse** : transcription, interprétation et détection d’émotion en un clic.  
4. **Génération d’image** : visionnez une illustration unique de votre rêve.  
5. **Historique** : parcourez vos rêves passés avec leurs transcriptions et analyses.

---

## ⚙️ Installation

```bash
# Clonez le dépôt
git clone https://github.com/coffeeaddictt/dream_interpreter.git
cd dream_interpreter

# Créez un environnement Python
python3 -m venv venv
source venv/bin/activate

# Installez les dépendances
pip install -r requirements.txt
