# Blueprint - Système RAG Hybride GreenPower Solutions

Groupe 2- Fabien G, Dorian, Paul, Noha, Sacha

## 📋 Vue d'ensemble du projet

Système de **Retrieval-Augmented Generation (RAG) Hybride** qui combine deux approches complémentaires pour répondre intelligemment aux questions sur les produits et événements de GreenPower Solutions :

- **Recherche vectorielle** (Qdrant) pour les questions descriptives simples
- **Graphe de connaissances** (Neo4j) pour les questions relationnelles complexes nécessitant du multi-hop reasoning

### 🎯 Objectif

Permettre de poser des questions en langage naturel sur :
- Les produits solaires autonomes GreenPower
- Les événements (salons, festivals alimentés)
- Les ventes et déploiements
- Les projets R&D
- Les relations entre ces entités

### 🏗️ Architecture Globale

```
┌──────────────────────────────────────────────────────────────────────┐
│                      INTERFACE UTILISATEUR                            │
│                     (app_hybrid.py - Streamlit)                       │
│                                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐        │
│  │   RAG    │  │   RAG    │  │ Routeur  │  │  Dashboard   │        │
│  │  Simple  │  │ Hybride  │  │  Intel.  │  │  Métriques   │        │
│  │ (Qdrant) │  │  Q+Neo4j │  │  (Auto)  │  │ (dashboard.py)│        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE DE ROUTAGE                         │
│                   (hybrid_rag.py)                            │
│                                                               │
│  classify_question() → détermine la stratégie optimale       │
│     ├── Keywords multi-hop → RAG Hybride                     │
│     └── Keywords simples → RAG Simple                        │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
┌──────────────────┐                    ┌──────────────────┐
│  QDRANT          │                    │  NEO4J           │
│  (Vectoriel)     │                    │  (Graphe)        │
│                  │                    │                  │
│ • Embeddings     │                    │ • Nœuds:         │
│ • Similarité     │                    │   - Product      │
│   sémantique     │                    │   - Event        │
│ • Documents      │                    │   - TradeShow    │
│                  │                    │   - Sale         │
│                  │                    │   - RDProject    │
│                  │                    │   - BatteryType  │
│                  │                    │                  │
│                  │                    │ • Relations:     │
│                  │                    │   - DEPLOYED_AT  │
│                  │                    │   - SOLD_AT      │
│                  │                    │   - USES_BATTERY │
│                  │                    │   - etc.         │
└──────────────────┘                    └──────────────────┘
        │                                         │
        └────────────────────┬────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     MISTRAL AI LLM                           │
│  • Génération de réponses contextualisées                    │
│  • Embeddings (mistral-embed)                                │
│  • Chat (mistral-small-latest)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Modules et Fonctionnalités

### 1. **hybrid_rag.py** - Routeur Intelligent et Orchestrateur

**Rôle principal**: Classifier les questions et orchestrer la stratégie de réponse appropriée.

#### Classe `HybridRAG`

**Fonctionnalités principales**:

- **`classify_question(question)`**
  - Analyse la question pour déterminer la complexité
  - Détecte les mots-clés multi-hop (relations, agrégations, événements)
  - Détecte les mots-clés simples (descriptions, spécifications)
  - Retourne: `"simple"` ou `"multi_hop"`

- **`query_simple(question, vector_store)`**
  - RAG classique avec Qdrant uniquement
  - Recherche par similarité sémantique (top 3 documents)
  - Génère une réponse avec le LLM Mistral
  - Cas d'usage: "Qu'est-ce que le produit GreenPower Max?"

- **`query_hybrid(question, vector_store)`**
  - RAG hybride combinant Qdrant + Neo4j
  - Récupère le contexte vectoriel (Qdrant)
  - Récupère le contexte relationnel (Neo4j)
  - Fusionne les deux contextes dans un prompt enrichi
  - Cas d'usage: "Quels événements ont utilisé des produits vendus à Pollutec Paris?"

- **`query(question, vector_store, force_strategy=None)`**
  - Point d'entrée principal avec routage automatique
  - Permet de forcer une stratégie si nécessaire
  - Retourne: `{answer, sources, strategy}`

- **`explain_routing(question)`**
  - Explique pourquoi une stratégie a été choisie
  - Utile pour la transparence et le debugging

**Exemples de questions traitées**:
- Simple: "Quel est le prix du GreenPower Max?"
- Multi-hop: "Quels projets R&D visent les produits utilisés aux festivals?"
- Multi-hop: "Quel est le CO2 total économisé par PG-M01?"

---

### 2. **neo4j_query.py** - Requêtes Multi-Hop sur le Graphe

**Rôle principal**: Exécuter des requêtes Cypher complexes pour extraire des informations relationnelles.

#### Classe `Neo4jQuerier`

**Requêtes multi-hop complexes**:

- **`query_events_with_products_sold_at_tradeshows(location=None)`**
  - Chemin: `Product → DEPLOYED_AT → Event` + `Product ← INCLUDES_PRODUCT ← Sale → SOLD_AT → TradeShow`
  - Question: "Quels événements ont utilisé des produits vendus à Pollutec Paris?"
  - Multi-hop: 2 sauts dans le graphe

- **`query_total_co2_saved_by_product(product_id)`**
  - Chemin: `Product → DEPLOYED_AT → Event` (avec agrégation CO2)
  - Question: "Quel est le CO2 total économisé par tous les déploiements de PG-M01?"
  - Agrégation: Somme des réductions CO2 sur tous les événements

- **`query_tradeshows_sales_by_customer_type(customer_type)`**
  - Chemin: `TradeShow ← SOLD_AT ← Sale` (filtré par type client)
  - Question: "Quels salons ont généré le plus de revenus avec les collectivités?"
  - Agrégation: Somme des revenus par salon

- **`query_rd_projects_for_festival_products()`**
  - Chemin: `Event ← DEPLOYED_AT ← Product ← TARGETS_PRODUCT ← RDProject`
  - Question: "Quels projets R&D visent les produits utilisés aux festivals?"
  - Multi-hop: 3 sauts + filtrage par type d'événement

**Requêtes simples**:

- **`query_products_by_battery_type(battery_type)`**
  - Recherche produits par type de batterie (LiFePO4, Tesla)

- **`query_top_revenue_tradeshows(limit=5)`**
  - Top salons par revenu généré

- **`query_product_sales_across_tradeshows(product_id)`**
  - Historique des ventes d'un produit sur tous les salons

**Fonctions utilitaires**:

- **`get_graph_context_for_question(question)`**
  - Analyse la question et exécute automatiquement les requêtes pertinentes
  - Détecte patterns: événements, CO2, collectivités, R&D, batteries, etc.
  - Retourne une liste de résultats structurés

- **`format_graph_context(context)`**
  - Formate les résultats du graphe en texte lisible pour le LLM
  - Convertit les résultats Neo4j en contexte textuel

---

### 3. **neo4j_loader.py** - Chargement des Données dans le Graphe

**Rôle principal**: Parser les fichiers JSON et créer le graphe Neo4j complet avec toutes les relations.

#### Classe `Neo4jLoader`

**Fonctionnalités**:

- **`clear_database()`**
  - Nettoie toutes les données du graphe
  - Utilisé avant un rechargement complet

- **`create_indexes()`**
  - Crée les index sur les IDs pour optimiser les performances
  - Index sur: product_id, event_id, trade_show_id, rd_project_id, customer_type

- **`load_products(products_file)`**
  - Parse `greenpower_products_enriched.json`
  - Crée les nœuds `Product` avec toutes leurs propriétés:
    - Spécifications techniques (puissance, batterie, solaire)
    - Prix et marges
    - Réduction CO2
  - Crée les nœuds `BatteryType` et les relations `USES_BATTERY`

- **`load_events(events_file)`**
  - Parse `greenpower_events_enriched.json`
  - Crée deux types d'événements:
    1. **TradeShows** (salons/expositions):
       - Nœuds `TradeShow` avec leads et revenus
       - Relations `DISPLAYED_AT` (produits affichés)
       - Crée des nœuds `Sale` par type de client (particuliers, entreprises, collectivités)
       - Relations `SOLD_AT` (ventes → salon)
       - Relations `INCLUDES_PRODUCT` avec quantités (ex: "PG-M01 x3")

    2. **Powered Events** (événements alimentés):
       - Nœuds `Event` (festivals, concerts)
       - Relations `DEPLOYED_AT` avec quantités de produits déployés
       - Propriétés: attendees, runtime, fuel_saved, co2_reduction

- **`load_rd_projects(rd_file)`**
  - Parse `greenpower_rd_innovations.json`
  - Crée les nœuds `RDProject` avec:
    - Status (Active, Testing, Deployed)
    - Objectif
    - Économies projetées
  - Relations `TARGETS_PRODUCT` vers les produits ciblés

- **`load_all()`**
  - Charge toutes les données en une seule commande
  - Séquence: clear → indexes → products → events → rd_projects

- **`verify_data()`**
  - Affiche les statistiques du graphe (nombre de nœuds et relations)
  - Utile pour vérifier que tout a été chargé correctement

**Modèle de données créé**:
- 66 nœuds au total
- 149 relations
- 6 types de nœuds: Product, Event, TradeShow, Sale, RDProject, BatteryType

---

### 4. **app_hybrid.py** - Interface Utilisateur Streamlit

**Rôle principal**: Interface web interactive pour interagir avec le système RAG hybride.

#### Fonctionnalités de l'application

**Architecture Streamlit**:

- **`init_components()`** (cached)
  - Initialise Qdrant client
  - Initialise Mistral embeddings
  - Initialise Mistral LLM
  - Initialise HybridRAG
  - Mise en cache pour éviter réinitialisations

- **`load_and_index_documents()`** (cached)
  - Charge tous les documents depuis `data/`
  - Supporte: .txt, .json, .csv, .pdf
  - Découpe en chunks (1000 chars, overlap 200)
  - Crée/charge la collection Qdrant
  - Mise en cache: instant après le premier chargement

**3 Onglets principaux**:

1. **RAG Classique (Qdrant)**
   - Utilise uniquement la recherche vectorielle
   - Affiche la réponse + sources Qdrant
   - Idéal pour: descriptions, spécifications, prix

2. **RAG+Graph Multi-Hop**
   - Force l'utilisation de Qdrant + Neo4j
   - Affiche la réponse + sources des deux bases
   - Vue séparée des sources vectorielles et graphe
   - Idéal pour: relations, agrégations, multi-hop

3. **Routeur Intelligent (Auto)**
   - Mode recommandé
   - Classification automatique de la question
   - Explique la décision du routeur
   - Badge visuel de la stratégie choisie
   - Sources adaptées à la stratégie

**Sidebar**:
- Bouton "🔄 Recharger" : réinitialise le cache Streamlit
- Bouton "🗑️ Réinitialiser" : supprime la collection Qdrant
- Bouton "📊 Charger Neo4j" : lance le chargement du graphe
- Exemples de questions simples et multi-hop

**Loaders de documents**:
- `load_txt()`: Fichiers texte
- `load_json()`: Fichiers JSON (formatés pour embedding)
- `load_csv()`: CSV (chaque ligne = 1 document)
- `load_pdf()`: PDF (chaque page = 1 document)

---

### 5. **init_system.py** - Script d'Initialisation

**Rôle principal**: Vérifier et initialiser tous les composants du système.

#### Fonctionnalités

- **`check_env()`**
  - Vérifie la présence de toutes les variables d'environnement requises
  - Masque les secrets dans l'affichage
  - Variables vérifiées: MISTRAL_API_KEY, QDRANT_*, NEO4J_*

- **`check_neo4j_connection()`**
  - Teste la connexion à Neo4j
  - Exécute une requête simple pour valider

- **`check_qdrant_connection()`**
  - Teste la connexion à Qdrant
  - Liste les collections disponibles

- **`load_neo4j_data()`**
  - Vérifie si des données existent déjà
  - Demande confirmation avant rechargement
  - Lance le chargement complet si nécessaire

- **`show_summary()`**
  - Affiche le résumé de l'initialisation
  - Commandes utiles
  - Conseils d'utilisation

**Workflow**:
1. Vérification des variables d'environnement
2. Test connexion Neo4j
3. Test connexion Qdrant
4. Chargement des données Neo4j (si nécessaire)
5. Affichage du résumé

---

### 6. **dashboard.py** - Dashboard de Métriques et Performance

**Rôle principal**: Collecter et afficher les métriques de performance en temps réel de Qdrant et Neo4j avec des visualisations graphiques.

#### Classe `DashboardMetrics`

**Fonctionnalités principales**:

- **`get_qdrant_metrics()`**
  - Récupère les statistiques de la base vectorielle Qdrant
  - Retourne: nombre de collections, documents indexés, vecteurs, dimension vectorielle
  - Détecte si la collection principale existe
  - Gestion d'erreur avec affichage des messages

- **`get_neo4j_metrics()`**
  - Récupère les statistiques du graphe Neo4j
  - Compte les nœuds par type (Product, Event, TradeShow, Sale, RDProject, BatteryType)
  - Compte les relations par type (DEPLOYED_AT, SOLD_AT, USES_BATTERY, etc.)
  - Retourne: total des nœuds, total des relations, distributions par type
  - Requêtes Cypher optimisées avec agrégations

- **`measure_qdrant_search_time(question, k=3)`**
  - Mesure le temps de recherche vectorielle Qdrant
  - Paramètres: question de test, nombre de résultats (k)
  - Retourne: temps en millisecondes, nombre de résultats trouvés
  - Utile pour benchmarking et optimisation

- **`measure_neo4j_query_time(query_func, *args)`**
  - Mesure le temps d'exécution d'une requête Neo4j
  - Accepte n'importe quelle fonction de requête Neo4j
  - Retourne: temps en millisecondes, nombre de résultats
  - Permet de comparer les performances de différentes requêtes

#### Fonction `render_dashboard(qdrant_client, neo4j_querier, vector_store)`

**Interface utilisateur du dashboard** (intégrée comme onglet dans app_hybrid.py):

**Section 1 - Métriques Qdrant** :
- Affichage des métriques en temps réel :
  - Nombre de collections
  - Documents indexés
  - Vecteurs stockés
  - Dimension vectorielle
- **Test de performance de recherche** :
  - Bouton "Lancer test Qdrant"
  - Teste 3 questions types
  - Affiche tableau des temps de réponse
  - Graphique en barres des temps par question
  - Calcul et affichage du temps moyen

**Section 2 - Métriques Neo4j** :
- Affichage des métriques du graphe :
  - Total de nœuds
  - Total de relations
- **Graphiques de distribution** :
  - Graphique 1 : Distribution des nœuds par type (bar chart)
  - Graphique 2 : Distribution des relations par type (bar chart)
- **Test de performance des requêtes** :
  - Bouton "Lancer test Neo4j"
  - Teste 4 types de requêtes :
    1. Requête simple (top salons)
    2. Requête multi-hop (événements avec produits vendus)
    3. Agrégation (ventes par type client)
    4. Multi-hop complexe (R&D festivals, 3 sauts)
  - Affiche tableau comparatif
  - Graphique en barres des temps par requête
  - Calcul du temps moyen

**Section 3 - Comparaison RAG Simple vs Hybride** :
- Input pour question de test personnalisée
- Bouton "Comparer"
- **Comparaison côte à côte** :
  - RAG Simple (Qdrant uniquement) : temps, nombre de sources
  - RAG Hybride (Qdrant + Neo4j) : temps, nombre de sources combinées
- Graphique comparatif des temps
- Analyse du différentiel de performance en pourcentage
- Recommandations basées sur les résultats

**Section 4 - État du Système** :
- Indicateurs de statut :
  - 🟢 Qdrant : Opérationnel / 🔴 Erreur
  - 🟢 Neo4j : Opérationnel / 🔴 Erreur
  - 🟢 Global : Système OK / 🔴 Problème détecté
- Timestamp de dernière mise à jour
- **Auto-refresh** : Option pour rafraîchir automatiquement toutes les 5 secondes

**Caractéristiques techniques** :
- Utilise uniquement Streamlit natif (pas de bibliothèques externes lourdes)
- Graphiques avec `st.bar_chart()` et `st.metric()`
- Dataframes pandas pour tableaux de données
- Gestion d'erreurs robuste avec messages utilisateur
- Performance mesurée en millisecondes (précision 2 décimales)
- Design responsive avec colonnes Streamlit

**Cas d'usage** :
1. **Monitoring** : Surveiller l'état du système en temps réel
2. **Debugging** : Identifier les requêtes lentes
3. **Optimisation** : Comparer les performances des stratégies RAG
4. **Démo** : Montrer les capacités du système aux stakeholders
5. **Benchmarking** : Mesurer l'impact des changements d'infrastructure

---

## 🔄 Flux de Données

### Flux Question → Réponse (RAG Simple)

```
1. Utilisateur pose une question: "Quel est le prix du GreenPower Max?"
   ↓
2. HybridRAG.classify_question() → "simple"
   ↓
3. HybridRAG.query_simple()
   ↓
4. Recherche vectorielle Qdrant (top 3 documents)
   ↓
5. Contexte envoyé au LLM Mistral
   ↓
6. Génération de la réponse
   ↓
7. Retour: {answer, sources: {vector_docs}, strategy: "simple"}
```

### Flux Question → Réponse (RAG Hybride)

```
1. Utilisateur pose: "Quels événements ont utilisé des produits vendus à Paris?"
   ↓
2. HybridRAG.classify_question() → "multi_hop"
   ↓
3. HybridRAG.query_hybrid()
   ↓
4. Parallèlement:
   ├─> Qdrant: recherche vectorielle (top 3 docs)
   └─> Neo4j: get_graph_context_for_question()
       └─> Détecte: "événements" + "vendus" + "paris"
       └─> Exécute: query_events_with_products_sold_at_tradeshows("Paris")
   ↓
5. Fusion des deux contextes dans un prompt enrichi
   ↓
6. LLM Mistral génère réponse basée sur les deux sources
   ↓
7. Retour: {answer, sources: {vector_docs, graph_context}, strategy: "hybrid"}
```

---

## 💾 Modèle de Données Neo4j

### Nœuds

```
(Product)
├─ product_id
├─ name
├─ category
├─ continuous_power
├─ peak_power
├─ battery_capacity
├─ battery_type
├─ solar_capacity
├─ total_cost
├─ avg_selling_price
├─ margin_percentage
├─ co2_reduction
└─ rental_available

(Event)
├─ event_id
├─ name
├─ type (festival, concert, etc.)
├─ location
├─ date
├─ attendees
├─ runtime
├─ fuel_saved
└─ co2_reduction

(TradeShow)
├─ event_id
├─ name
├─ type
├─ location
├─ date
├─ leads_generated
└─ total_sales

(Sale)
├─ sale_id
├─ customer_type (particuliers, entreprises, collectivites)
├─ units
└─ total_revenue

(RDProject)
├─ project_id
├─ name
├─ status
├─ objective
└─ projected_savings

(BatteryType)
└─ type (LiFePO4, Tesla, etc.)
```

### Relations

```
(Product)-[:DEPLOYED_AT {quantity}]->(Event)
(Product)-[:DISPLAYED_AT]->(TradeShow)
(Product)-[:USES_BATTERY]->(BatteryType)

(Sale)-[:SOLD_AT]->(TradeShow)
(Sale)-[:INCLUDES_PRODUCT {quantity}]->(Product)

(RDProject)-[:TARGETS_PRODUCT]->(Product)
```

---

## 🎯 Cas d'Usage Typiques

### 1. Questions Simples (RAG Vectoriel)

**Question**: "Qu'est-ce que le GreenPower Max?"
- **Stratégie**: RAG Simple
- **Source**: Qdrant uniquement
- **Pourquoi**: Question descriptive, pas de relations

**Question**: "Quels sont les produits avec batteries LiFePO4?"
- **Stratégie**: RAG Hybride
- **Source**: Neo4j (query_products_by_battery_type)
- **Pourquoi**: Requête structurée simple sur le graphe

### 2. Questions Multi-Hop (RAG Hybride)

**Question**: "Quels événements ont utilisé des produits vendus à Pollutec Paris?"
- **Stratégie**: RAG Hybride
- **Chemin**: Product → DEPLOYED_AT → Event + Product ← INCLUDES_PRODUCT ← Sale → SOLD_AT → TradeShow
- **Requête**: query_events_with_products_sold_at_tradeshows("Paris")
- **Complexité**: 2-3 sauts dans le graphe

**Question**: "Quels projets R&D visent les produits utilisés aux festivals?"
- **Stratégie**: RAG Hybride
- **Chemin**: RDProject → TARGETS_PRODUCT → Product → DEPLOYED_AT → Event (type=festival)
- **Requête**: query_rd_projects_for_festival_products()
- **Complexité**: 3 sauts + filtrage

### 3. Agrégations Complexes

**Question**: "Quel est le CO2 total économisé par le produit PG-M01?"
- **Stratégie**: RAG Hybride
- **Opération**: Somme de tous les déploiements
- **Requête**: query_total_co2_saved_by_product("PG-M01")
- **Type**: Agrégation avec calcul

**Question**: "Quels salons ont généré le plus de revenus avec les collectivités?"
- **Stratégie**: RAG Hybride
- **Opération**: Groupement + agrégation + tri
- **Requête**: query_tradeshows_sales_by_customer_type("collectivites")
- **Type**: Agrégation multi-nœuds

---

## 🚀 Déploiement et Configuration

### Variables d'environnement requises (.env)

```bash
# Mistral AI
MISTRAL_API_KEY=your_key_here

# Qdrant
QDRANT_ENDPOINT=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_key_here

# Neo4j
NEO4J_URI=neo4j+s://your-instance.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

### Commandes principales

```bash
# Installation complète
./deploy.sh

# Initialisation manuelle
python init_system.py

# Recharger Neo4j
python neo4j_loader.py

# Tester requêtes Neo4j
python neo4j_query.py

# Lancer l'application
streamlit run app_hybrid.py
```

---

## 🔍 Avantages de l'Architecture Hybride

### RAG Simple (Qdrant seul)
✅ Rapide
✅ Bon pour descriptions et spécifications
✅ Pas de setup complexe
❌ Ne comprend pas les relations
❌ Pas d'agrégations

### RAG Hybride (Qdrant + Neo4j)
✅ Multi-hop reasoning
✅ Agrégations complexes (sommes, moyennes, totaux)
✅ Comprend les relations entre entités
✅ Répond à "Qui?", "Où?", "Combien?", "Quels événements avec...?"
⚠️ Plus lent (2 requêtes)
⚠️ Nécessite un graphe bien structuré

### Routeur Intelligent
✅ Choisit automatiquement la meilleure stratégie
✅ Optimise performance (évite Neo4j si inutile)
✅ Transparent pour l'utilisateur
✅ Explicable (explain_routing)

---

## 🛠️ Technologies Utilisées

| Technologie | Rôle |
|------------|------|
| **Streamlit** | Interface web interactive |
| **Qdrant** | Base vectorielle (embeddings) |
| **Neo4j** | Graphe de connaissances |
| **Mistral AI** | LLM + Embeddings |
| **LangChain** | Framework RAG |
| **Python 3.9+** | Langage principal |
