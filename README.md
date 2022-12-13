# Socratext: OCR et extraction d'informations de documents administratifs 

Ce projet est développé dans le cadre du [programme 10%](https://10pourcent.etalab.studio/) piloté par Etalab. 
Lors des ateliers de lancement qui se sont tenus les 13 et 14 juin, des premiers éléments de cadrage ont été rassemblés [ici](https://github.com/etalab-ia/programme10pourcent/wiki/Ateliers-OCR-et-extraction-d'informations-%C3%A0-partir-de-documents-administratifs).

Les administrations ont régulièrement besoin d’exploiter en masse des documents administratifs sous des formats non directement exploitables (PDF scannés, images, etc.). L’information contenue dans ces documents, pour être exploitée, doit passer par des étapes d'OCR, d’extraction et de structuration de l’information, qui est vite très chronophage si elle doit être réalisée à la main. 

Ce repertoire a pour objectif de co-construire des solutions d'OCR, d'extratction d'informations et de compréhension de documents (extraire la structure d'un document), et ce en prenant en compte les différents besoins rencontrés par les administrations participant au programme. 

## Extraction d'information de photos de tickets de caisse

### Entraînement d'un modèle LayoutLMv2

Sur le [SSP Cloud](https://datalab.sspcloud.fr/home), lancer [ce service](https://datalab.sspcloud.fr/launcher/ide/vscode-python-gpu?autoLaunch=true&security.allowlist.enabled=false&service.image.pullPolicy=«Always»&onyxia.friendlyName=«vscode-python-gpu-pull») (configuration actuelle : 1 GPU). Installer les librairies de `requirements.txt` et run le script `setup.sh`. Pour lancer l'entraînement d'un modèle, et envoyer les logs sur l'espace de stockage du SSP Cloud :

```
python src/train.py --s3
```

Les flags `--lr` et `--batch-size` permettent de spécifier le pas d'apprentissage et la taille des batchs respectivement, par exemple :

```
python src/train.py --s3 --lr 0.004 --batch-size 5
```

Pour lancer `Tensorboard`, run le script `tensorboard.sh`.


