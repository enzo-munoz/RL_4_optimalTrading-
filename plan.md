PRD : Système de Trading Algorithmique par Apprentissage par Renforcement Profond (Sections 1-5)

1. Vue d'ensemble du projetL'objectif est de reproduire une stratégie de trading optimal exploitant des informations latentes (cachées) via un mélange de Deep Reinforcement Learning (DRL) et de Réseaux de Neurones Récurrents (RNN/GRU). Le système doit apprendre à naviguer dans un marché dont les paramètres (moyenne, vitesse de retour, volatilité) changent selon des régimes de Markov non observés.

2. Spécifications de l'Environnement (RL Setup)

2.1 Modèle de Marché (Signal de Trading)
L'environnement doit simuler un processus d'Ornstein-Uhlenbeck (OU) à changement de régime :
Équation SDE : 
$dS_t = \kappa_t(\theta_t - S_t)dt + \sigma_t dW_t$Paramètres Latents : $\theta_t$ (moyenne), $\kappa_t$ (vitesse), $\sigma_t$ (volatilité).

Dynamique des régimes : Les paramètres suivent une chaîne de Markov discrète.Discrétisation temporelle : 
$\Delta t = 0.2$ (Pas de temps d'Euler-Maruyama pour la simulation).

2.2 Espace des États (Observations)
L'agent reçoit à chaque pas de temps $t$ :
Le niveau actuel du signal $S_t$.L'inventaire actuel $I_t$.Une fenêtre historique de taille $W$ des prix passés : $\{S_{t-W}, \dots, S_t\}$.

2.3 Espace des ActionsAction 
($a_t$) : L'agent décide du prochain niveau d'inventaire cible $I_{t+1}$.Contrainte : $I \in [-10, 10]$ (Espace continu).

2.4 Fonction de Récompense (Reward)

La récompense immédiate $r_t$ est définie par le profit PnL moins les coûts de transaction :
$$r_t = I_{t+1}(S_{t+1} - S_t) - \lambda |I_{t+1} - I_t|$$Où $\lambda$ représente le coefficient de coût de transaction.

3. Architectures à Implémenter (DDPG + GRU)

Le développeur doit implémenter trois variantes basées sur le Deep Deterministic Policy Gradient (DDPG) :hid-DDPG (One-step) :Un GRU traite la fenêtre $W$.L'état caché du GRU est concaténé à l'inventaire $I_t$ pour alimenter l'Acteur et le Critique.prob-DDPG (Two-step) :Un GRU est pré-entraîné pour prédire les probabilités de régime de $\theta_t$ (Classification).
Ces probabilités deviennent les entrées de l'agent RL.reg-DDPG (Two-step) :Un GRU est pré-entraîné pour prédire le prix suivant $S_{t+1}$ (Régression).La prédiction est utilisée comme feature par l'agent RL.

4. Règles de Développement pour l'Agent IA. 
L'agent de génération de code doit respecter les règles strictes suivantes :

Règle 1 : Target Networks. Utiliser des réseaux "cibles" (Target Actor/Critic) avec une mise à jour douce (soft update) pour stabiliser l'apprentissage.

Règle 2 : Normalisation. Toutes les entrées (prix, inventaire) doivent être normalisées avant d'entrer dans les réseaux de neurones.

Règle 3 : Replay Buffer. Implémenter un tampon d'expérience (Experience Replay) pour stocker les transitions $(s_t, a_t, r_t, s_{t+1})$.

Règle 4 : Exploration. Ajouter un bruit (Ornstein-Uhlenbeck ou Gaussien avec décroissance) à l'action de l'acteur pendant la phase d'entraînement.

Règle 5 : Modularité. Séparer la logique de la simulation du signal (Environment) de la logique de l'agent (Model).

5. Paramètres Techniques et Hyperparamètres

Paramètre,Valeur suggérée
Fenêtre d'observation (W), 10 ou 20
Nombre d'épisodes, 10 000
Taille du Batch, 512
Optimiseur, Adam
Learning Rate (LR), 10−3
Facteur de remise (γ), 1.0 (ou proche de 1 pour le trading)
Taille de l'inventaire max, 10 unités

6. Bonnes Pratiques pour l'Environnement de RL

Interface Gym/Gymnasium : Créer une classe héritant de gym.Env pour assurer la compatibilité avec les bibliothèques standards.

Gestion des épisodes : Chaque épisode doit correspondre à une trajectoire de signal de longueur fixe (ex: $T=500$).

Reproductibilité : Fixer les graines aléatoires (seeds) pour NumPy, PyTorc het l'environnement de marché.

Logging : Suivre le profit cumulé, les coûts totaux et l'évolution des Q-values pendant l'entraînement.

Évaluation : Tester l'agent sur des trajectoires de signaux que l'agent n'a pas "vues" pendant l'entraînement (Out-of-sample).