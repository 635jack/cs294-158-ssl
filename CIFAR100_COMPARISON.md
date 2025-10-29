# Comparaison CIFAR10 vs CIFAR100

## Modifications effectuées

### 1. Support de CIFAR100 dans le code ✅

Les fichiers suivants ont été modifiés pour supporter CIFAR100 :
- `deepul_helper/data.py` : ajout de CIFAR100 dans get_transform() et get_datasets()
- `deepul_helper/tasks/rotation.py` : support de CIFAR100
- `deepul_helper/tasks/simclr.py` : support de CIFAR100
- `deepul_helper/tasks/context_encoder.py` : aucune modification nécessaire (fonctionne déjà)

### 2. Scripts d'entraînement créés ✅

Nouveaux scripts dans `run/` :
- `run_cifar100_context_encoder.sh`
- `run_cifar100_rotation.sh`
- `run_cifar100_simclr.sh`

## Pour entraîner les modèles CIFAR100

```bash
# Context Encoder
./run/run_cifar100_context_encoder.sh

# Rotation Prediction
./run/run_cifar100_rotation.sh

# SimCLR
./run/run_cifar100_simclr.sh
```

## Section à ajouter au notebook

Après la cellule de comparaison des résultats CIFAR10, ajouter :

### Nouvelle section markdown :
```markdown
# Comparaison CIFAR100

CIFAR100 contient 60 000 images (comme CIFAR10) mais avec 100 classes au lieu de 10.
C'est donc un problème de classification plus difficile.
```

### Code pour charger et évaluer les modèles CIFAR100 :

```python
# Charger les résultats CIFAR100
results_cifar100 = []

# Context Encoder CIFAR100
model, linear_classifier, train_loader, test_loader = load_model_and_data('context_encoder', 'cifar100')
evaluate_accuracy(model, linear_classifier, train_loader, test_loader)
train_acc1, train_acc5 = evaluate_classifier(model, linear_classifier, train_loader)
test_acc1, test_acc5 = evaluate_classifier(model, linear_classifier, test_loader)
results_cifar100.append({'Model': 'Context_encoder','train_acc1':train_acc1,'train_acc5':train_acc5,'test_acc1':test_acc1,'test_acc5':test_acc5})

# Rotation CIFAR100
model, linear_classifier, train_loader, test_loader = load_model_and_data('rotation', 'cifar100')
evaluate_accuracy(model, linear_classifier, train_loader, test_loader)
train_acc1, train_acc5 = evaluate_classifier(model, linear_classifier, train_loader)
test_acc1, test_acc5 = evaluate_classifier(model, linear_classifier, test_loader)
results_cifar100.append({'Model': 'Rotation Prediction','train_acc1':train_acc1,'train_acc5':train_acc5,'test_acc1':test_acc1,'test_acc5':test_acc5})

# SimCLR CIFAR100
model, linear_classifier, train_loader, test_loader = load_model_and_data('simclr', 'cifar100')
evaluate_accuracy(model, linear_classifier, train_loader, test_loader)
train_acc1, train_acc5 = evaluate_classifier(model, linear_classifier, train_loader)
test_acc1, test_acc5 = evaluate_classifier(model, linear_classifier, test_loader)
results_cifar100.append({'Model': 'SimCLR','train_acc1':train_acc1,'train_acc5':train_acc5,'test_acc1':test_acc1,'test_acc5':test_acc5})
```

### Visualisation comparative CIFAR10 vs CIFAR100 :

```python
import pandas as pd
import matplotlib.pyplot as plt

df_cifar10 = pd.DataFrame(results)
df_cifar100 = pd.DataFrame(results_cifar100)

# Comparaison côte à côte
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# CIFAR10
axes[0].bar(df_cifar10['Model'], df_cifar10['test_acc1'], color='steelblue')
axes[0].set_xlabel('Modèles')
axes[0].set_ylabel('Test Top-1 Accuracy')
axes[0].set_title('CIFAR10 - Test Accuracy')
axes[0].set_ylim([0, 100])

# CIFAR100
axes[1].bar(df_cifar100['Model'], df_cifar100['test_acc1'], color='coral')
axes[1].set_xlabel('Modèles')
axes[1].set_ylabel('Test Top-1 Accuracy')
axes[1].set_title('CIFAR100 - Test Accuracy')
axes[1].set_ylim([0, 100])

plt.tight_layout()
plt.show()

# Tableau comparatif
comparison = pd.DataFrame({
    'Model': df_cifar10['Model'],
    'CIFAR10_test_acc1': df_cifar10['test_acc1'],
    'CIFAR100_test_acc1': df_cifar100['test_acc1'],
    'Diff': df_cifar10['test_acc1'] - df_cifar100['test_acc1']
})
print(comparison)
```

## Résultats attendus

On s'attend à ce que :
- Les performances sur CIFAR100 soient inférieures à CIFAR10 (plus de classes)
- SimCLR et Rotation Prediction restent les meilleures méthodes
- La baisse de performance soit similaire pour toutes les méthodes

## Datasets comparés

| Dataset | Images | Classes | Taille | Difficulté |
|---------|--------|---------|--------|------------|
| CIFAR10 | 60 000 | 10 | ~170 MB | Moyenne |
| CIFAR100 | 60 000 | 100 | ~170 MB | Difficile |

