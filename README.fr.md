[🇫🇷 Français](README.fr.md) | [🇬🇧 English](README.md)

# Réseau de neurones from scratch en C

Un petit réseau de neurones à propagation avant, écrit en C pur, qui apprend à reconnaître des chiffres manuscrits à partir du jeu de données **MNIST**. Tout est construit de zéro : la bibliothèque de matrices, les passes avant et arrière, la boucle d'entraînement ainsi que la sauvegarde et le chargement des réseaux entraînés. La seule dépendance est la bibliothèque standard du C (plus `libm`).

## Fonctionnalités

- Réseau entièrement connecté avec une couche cachée : **784 → 500 → 10** (configurable)
- Activation ReLU sur la couche cachée et **softmax** numériquement stable sur la couche de sortie
- Entraînement par descente de gradient stochastique (une image par mise à jour) avec rétropropagation
- Sauvegarde et chargement d'un réseau entraîné dans un dossier de fichiers texte
- `MatrixLib`, une bibliothèque de matrices maison, pour toute l'algèbre linéaire
- Chargement des images MNIST directement depuis des fichiers CSV

## Structure du projet

```
neural-network/
├── Makefile
├── main.c                 Point d'entrée : c'est le fichier à modifier pour entraîner ou tester un réseau
├── parameters.h           Taille du réseau, taux d'apprentissage, chemins des données et des sauvegardes
├── MatrixLib/
│   ├── MatrixLib.h        Type Matrix et opérations (déclarations)
│   └── MatrixLib.c        Implémentation des matrices
├── network/
│   ├── neural_network.h/c Création, entraînement, prédiction, sauvegarde et chargement du réseau
│   └── activations.h/c    ReLU, softmax, sigmoïde
├── util/
│   └── img.h/c            Chargeur CSV MNIST et affichage des images
├── data/
│   ├── mnist_train.csv    Jeu d'entraînement (60 000 images)
│   └── mnist_test.csv     Jeu de test (10 000 images)
├── network_save/          Un réseau sauvegardé (créé par le programme)
└── exe/main.exe           Programme compilé (créé par `make`)
```

Le projet utilise une **compilation unitaire** (« unity build ») : `main.c` inclut les autres fichiers `.c`, de sorte que seul `main.c` est compilé. Il ne faut donc pas compiler les autres fichiers `.c` séparément.

## Prérequis

- Un compilateur C (`gcc` est utilisé par défaut)
- `make`
- Le jeu de données MNIST au format CSV, placé dans `data/` (voir ci-dessous)

## Jeu de données

Le programme attend deux fichiers CSV :

```
data/mnist_train.csv
data/mnist_test.csv
```

Chaque fichier commence par une ligne d'en-tête, puis contient une image par ligne : d'abord l'étiquette du chiffre (de 0 à 9), puis les 784 valeurs de pixels (de 0 à 255) de l'image 28×28. Les valeurs des pixels sont ramenées à l'intervalle `[0, 1]` lors du chargement. Si vous n'avez pas ces fichiers, le jeu MNIST au format CSV est facilement disponible en ligne (par exemple sur Kaggle).

## Compilation et exécution

```bash
make          # compile le programme vers ./exe/main.exe
make run      # compile si nécessaire, puis lance le programme
make clean    # supprime l'exécutable
```

Lancez toutes ces commandes depuis la racine du projet, car les chemins des données et des sauvegardes définis dans `parameters.h` sont relatifs au dossier de travail.

## Utilisation

Le comportement du programme est défini par `main.c`. Ce fichier contient deux blocs : l'un est actif, l'autre est commenté. Pour passer de l'un à l'autre, déplacez les marqueurs de commentaire `/* ... */`.

### 1. Créer et tester un nouveau réseau (mode par défaut)

Le bloc actif crée un réseau avec des poids aléatoires, mesure sa précision sur les images de test, puis le sauvegarde dans `network_save/` :

```c
NeuralNetwork* net = network_create(INPUT, HIDDEN, OUTPUT, LEARNING_RATE);
Img** imgs_test = csv_to_imgs(TEST_DATASET_PATH, 0, N_TEST_IMGS);
printf("Accuracy : %1.1f %c\n", 100*network_predict_imgs(net, imgs_test, N_TEST_IMGS), '%');
network_save(net, LOAD_DIR_PATH);
```

Un réseau non entraîné obtient une précision proche du hasard (environ 10 %). Lancez d'abord ce mode : il produit le réseau de départ que l'étape d'entraînement charge ensuite.

### 2. Entraîner un réseau

Pour entraîner, mettez en commentaire le premier bloc et décommentez le second :

```c
Img** imgs_train = csv_to_imgs(TRAIN_DATASET_PATH, 0, N_TRAIN_IMGS);
Img** imgs_test = csv_to_imgs(TEST_DATASET_PATH, 0, N_TEST_IMGS);
NeuralNetwork* net = network_load(LOAD_DIR_PATH);
network_train_batch_imgs(net, imgs_train, N_TRAIN_IMGS);
printf("Accuracy : %1.1f %c\n", 100*network_predict_imgs(net, imgs_test, N_TEST_IMGS), '%');
network_save(net, SAVE_DIR_PATH);
```

Ce bloc charge le réseau depuis `network_save/`, l'entraîne une fois sur les `N_TRAIN_IMGS` premières images d'entraînement, affiche la précision obtenue sur les images de test, puis enregistre le résultat dans `network_save2/`. Avec les réglages par défaut (2 000 images d'entraînement et 1 000 images de test), un seul passage atteint environ 87 % de précision.

Pour continuer à améliorer un réseau, faites pointer `LOAD_DIR_PATH` vers le dossier de la dernière sauvegarde (ou répétez l'appel à `network_train_batch_imgs` pour effectuer plusieurs passages), et utilisez un `SAVE_DIR_PATH` différent, sans quoi vous écraseriez le réseau que vous venez de charger.

## Configuration

Tous les réglages se trouvent dans `parameters.h` :

| Constante | Valeur par défaut | Signification |
|-----------|-------------------|---------------|
| `INPUT` | 784 | Taille de la couche d'entrée (28×28 pixels) |
| `HIDDEN` | 500 | Taille de la couche cachée |
| `OUTPUT` | 10 | Taille de la couche de sortie (chiffres de 0 à 9) |
| `LEARNING_RATE` | 0.02 | Pas de la descente de gradient |
| `N_TRAIN_IMGS` | 2000 | Nombre d'images d'entraînement utilisées |
| `N_TEST_IMGS` | 1000 | Nombre d'images de test utilisées |
| `TRAIN_DATASET_PATH` / `TEST_DATASET_PATH` | `./data/mnist_*.csv` | Emplacement des jeux de données |
| `LOAD_DIR_PATH` | `network_save` | Dossier depuis lequel un réseau est chargé (et dossier de sauvegarde par défaut en mode test) |
| `SAVE_DIR_PATH` | `network_save2` | Dossier dans lequel un réseau entraîné est sauvegardé |

Les jeux complets contiennent 60 000 images d'entraînement et 10 000 images de test : vous pouvez donc augmenter `N_TRAIN_IMGS` et `N_TEST_IMGS` jusqu'à ces valeurs. Le temps d'entraînement croît linéairement avec le nombre d'images.

## Format d'un réseau sauvegardé

Un réseau est sauvegardé sous la forme d'un dossier de fichiers texte :

| Fichier | Contenu |
|---------|---------|
| `descriptor` | Taille de la couche d'entrée, de la couche cachée, de la couche de sortie, puis taux d'apprentissage, à raison d'une valeur par ligne |
| `hidden_weights`, `hidden_biases` | Paramètres de la couche cachée |
| `output_weights`, `output_biases` | Paramètres de la couche de sortie |

Chaque fichier de matrice commence par une ligne `lignes colonnes`, suivie d'une ligne par rangée de la matrice. Lors du chargement d'un réseau, les dimensions des matrices sont vérifiées par rapport au descripteur.

## Fonctionnement

Pour une image d'entraînement (un vecteur colonne `A0` de taille 784×1) et son étiquette one-hot `Y` :

1. **Passe avant :** `Z1 = W1·A0 + b1`, `A1 = ReLU(Z1)`, `Z2 = W2·A1 + b2`, `A2 = softmax(Z2)`
2. **Passe arrière :** `dZ2 = A2 − Y`, `dW2 = dZ2·A1ᵀ`, `dZ1 = (W2ᵀ·dZ2) ⊙ ReLU′(Z1)`, `dW1 = dZ1·A0ᵀ`, ainsi que les gradients des biais correspondants
3. **Mise à jour :** chaque paramètre est diminué du produit du taux d'apprentissage (`learning_rate`) par son gradient

Les poids et les biais sont initialisés avec des valeurs aléatoires uniformes dans `[-1/√n, 1/√n]`. Le générateur aléatoire n'est pas initialisé avec une graine variable : un réseau fraîchement créé est donc identique d'une exécution à l'autre.

### MatrixLib

`MatrixLib` fournit le type `Matrix` (`entries[i][j]`, `rows`, `cols`) et les opérations dont le réseau a besoin : `add`, `subtract`, `multiply` (produit terme à terme), `dot` (produit matriciel), `scale`, `transpose`, `apply`, `sum` et `matrix_argmax`, ainsi que l'affichage, la sauvegarde et le chargement. Ces opérations renvoient des matrices nouvellement allouées, qu'il faut libérer avec `matrix_free`, et elles interrompent le programme avec un message d'erreur lorsque les dimensions ne correspondent pas.

## Remarques

- L'exécutable `exe/main.exe` est recompilé par `make` dès qu'un fichier source ou un fichier d'en-tête est modifié.
- Le dossier `data/` est volumineux (environ 128 Mo) : il vaut mieux l'exclure du contrôle de version.
- L'entraînement effectue une mise à jour par image, ce qui est simple mais pas la méthode la plus rapide. Passer aux mini-batches constituerait une amélioration naturelle.