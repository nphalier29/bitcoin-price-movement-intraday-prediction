# Analyse intra-day du marché crypto

L’objectif de ce projet est de développer, en Python, un algorithme capable d’analyser l’impact de différentes variables quantitatives fréquemment utilisées en trading (telles que le RSI, le MACD, etc.) sur la fluctuation intra-day des cryptomonnaies.

Une approche orientée objet (POO) est retenue afin de permettre la réutilisation et le déploiement potentiel de la stratégie sur différentes cryptomonnaies. Cette architecture facilite également l’ajout futur de nouvelles variables ou indicateurs, ainsi que la modularité nécessaire pour tester, modifier ou intégrer de nouveaux modèles de Machine Learning.

L’objectif final est de générer un signal clair et minimaliste permettant d’anticiper si le Bitcoin va monter ou descendre à t+1.

## Processus de collaboration pour le projet (Mac et Windows)

### Step 1 : Ouvrir le terminal
- **Mac** : Terminal  
- **Windows** : PowerShell (ou Terminal intégré de VSCode)  

Se placer dans le dossier du projet :  
```bash
cd chemin/vers/le/projet
````

---

### Step 2 : Mettre à jour le projet

Récupérer les dernières modifications depuis GitHub :

```bash
git pull
```

---

### Step 3 : Créer et activer le venv

> Pour créer un venv.

```powershell
python -m venv venv
```
> Pour activer le venv dans vscode.

* **Windows** :

```powershell
.\venv\Scripts\activate
```

* **Mac / Linux** :

```bash
source venv/bin/activate
```

> Si l’activation fonctionne, `(venv)` apparaît au début de la ligne.
> Pour désactiver le venv plus tard :

```bash
deactivate
```

---

### Step 4 : Installer les dépendances

Si de nouveaux packages ont été ajoutés dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

> Toujours vérifier que le venv est actif avant d’installer des packages.

---

### Step 5 : Coder

* Modifier ou ajouter des fichiers Python.
* Pour l'ajout de nouveaux packages :

```bash
pip install nom_du_package
```

* Ensuite, mettre à jour `requirements.txt` pour que les autres aient les mêmes packages :

```bash
pip freeze > requirements.txt
```

---

### Step 6 : Sauvegarder le code

* Sauvegarder tes fichiers (`Ctrl + S` ou `Cmd + S`).
* Revenir dans le terminal et se placer dans le dossier du projet si nécessaire :

```bash
cd chemin/vers/le/projet
```

---

### Step 7 : Ajouter les fichiers au commit

* Ajouter tous les fichiers modifiés/nouveaux :

```bash
git add .
```

* Ou ajouter seulement certains fichiers :

```bash
git add nom_du_fichier
```

---

### Step 8 : Créer le commit

```bash
git commit -m "update (choisir sa propre phrase)"
```

---

### Step 9 : Pousser les modifications sur GitHub

* **Sur la branche principale (main)** :

```bash
git push origin main
```

* **Sur une branche séparée** :

```bash
git push origin nom_de_branche
```
