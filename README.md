# hybrid-owc-bem-validation
Protocole de modélisation validé pour structures flottantes hybrides OWC. Comparaison croisée Python / Capytaine / ANSYS Aqwa, mise en évidence des limites de la formulation BEM indirecte pour les chambres OWC, et benchmark de réduction du pitch (>90%) pour plateformes flottantes à dampers accordés.
# Protocole de modélisation pour structures flottantes hybrides OWC

Ce dépôt fournit un protocole de modélisation validé destiné aux chercheurs et ingénieurs
qui utilisent des outils tels que **Capytaine** ou **ANSYS Aqwa** pour l’étude de structures
flottantes hybrides intégrant des chambres OWC (Oscillating Water Column).

## Objectifs

- Proposer un protocole reproductible pour la modélisation de structures hybrides.
- Mettre en évidence les limites de la formulation indirecte de la méthode des éléments
  de frontière (BEM) appliquée aux chambres OWC.
- Valider un concept hybride innovant d’OWC et fournir un benchmark pour la conception
  de plateformes flottantes équipées de dampers accordés.
- Documenter une réduction de plus de 90 % du mouvement de pitch.

## Approche

Le projet s’appuie sur une **validation croisée rigoureuse** entre :

1. un modèle analytique Python à 14 degrés de liberté ;
2. Capytaine ;
3. ANSYS Aqwa.

Les scripts inclus permettent le post-traitement hydrodynamique (fichiers `.AH1`, `.LIS`),
la vérification des équations du modèle, et l’automatisation de graphiques (RAO, spectres).

## Impact

Ce travail comble un vide dans l’écosystème de l’énergie houlomotrice en fournissant
un protocole validé et un benchmark de conception pour les structures flottantes hybrides
à OWC. Il vise une soumission à une revue à comité de lecture (par ex. *Renewable Energy*
ou *Ocean Engineering*).

## Licence

[À compléter : MIT / Apache 2.0 / GPLv3]
