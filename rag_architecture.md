# Architecture Logique du RAG Hybride GreenPower

## Diagramme de Flux Principal

```mermaid
flowchart TB
    %% Nodes principaux
    User[👤 Utilisateur]
    App[🌿 Application Streamlit<br/>app_hybrid.py]
    Router[🧭 Routeur Intelligent<br/>HybridRAG.classify_question]

    %% Décisions
    Decision{Type de question?}

    %% Branches RAG
    SimpleRAG[📄 RAG Simple<br/>query_simple]
    HybridRAG[🔗 RAG Hybride<br/>query_hybrid]

    %% Sources de données
    Qdrant[(🗄️ Qdrant<br/>Vector Store)]
    Neo4j[(🔗 Neo4j<br/>Knowledge Graph)]

    %% Traitement
    VectorSearch[🔍 Recherche Vectorielle<br/>Similarité Sémantique]
    GraphQuery[🕸️ Requêtes Cypher<br/>Multi-hop Reasoning]

    %% Génération
    LLM[🤖 Mistral LLM<br/>Génération Réponse]

    %% Résultat
    Response[✨ Réponse Finale<br/>+ Sources]
    Dashboard[📊 Dashboard Métriques]

    %% Flux principal
    User -->|Question| App
    App --> Router
    Router --> Decision

    %% Branche Simple
    Decision -->|Simple<br/>Descriptive| SimpleRAG
    SimpleRAG --> VectorSearch
    VectorSearch --> Qdrant
    Qdrant -->|Documents<br/>Similaires| LLM

    %% Branche Hybride
    Decision -->|Multi-hop<br/>Relationnelle| HybridRAG
    HybridRAG --> VectorSearch
    HybridRAG --> GraphQuery
    VectorSearch --> Qdrant
    GraphQuery --> Neo4j

    Qdrant -->|Contexte<br/>Vectoriel| LLM
    Neo4j -->|Contexte<br/>Graphique| LLM

    %% Génération finale
    LLM --> Response
    Response --> User

    %% Monitoring
    Qdrant -.->|Métriques| Dashboard
    Neo4j -.->|Statistiques| Dashboard
    Dashboard -.-> User

    %% Styles
    classDef userClass fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef appClass fill:#10b981,stroke:#059669,color:#fff
    classDef routerClass fill:#9333ea,stroke:#6b21a8,color:#fff
    classDef simpleClass fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef hybridClass fill:#10b981,stroke:#059669,color:#fff
    classDef dbClass fill:#64748b,stroke:#334155,color:#fff
    classDef llmClass fill:#f59e0b,stroke:#d97706,color:#fff
    classDef responseClass fill:#ec4899,stroke:#be185d,color:#fff

    class User userClass
    class App,HybridRAG hybridClass
    class Router routerClass
    class SimpleRAG simpleClass
    class Qdrant,Neo4j dbClass
    class LLM llmClass
    class Response,Dashboard responseClass
```

## Architecture des Composants

```mermaid
graph TB
    subgraph Interface["🎨 Interface Utilisateur"]
        UI1[Tab 1: RAG Classique]
        UI2[Tab 2: RAG+Graph]
        UI3[Tab 3: Routeur Auto]
        UI4[Tab 4: Dashboard]
    end

    subgraph Core["⚙️ Logique Métier"]
        HR[HybridRAG<br/>hybrid_rag.py]
        NQ[Neo4jQuerier<br/>neo4j_query.py]
        NL[Neo4jLoader<br/>neo4j_loader.py]
    end

    subgraph Storage["💾 Stockage"]
        QDB[(Qdrant<br/>Collections)]
        NDB[(Neo4j<br/>Graph DB)]
        Files[📂 Data Files<br/>JSON/CSV/PDF]
    end

    subgraph External["🌐 Services Externes"]
        Mistral[Mistral AI API<br/>LLM + Embeddings]
    end

    %% Relations
    UI1 --> HR
    UI2 --> HR
    UI3 --> HR
    UI4 --> QDB
    UI4 --> NDB

    HR --> NQ
    HR --> QDB
    HR --> Mistral

    NQ --> NDB
    NL --> NDB
    NL --> Files

    Files --> QDB

    classDef interfaceClass fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef coreClass fill:#10b981,stroke:#059669,color:#fff
    classDef storageClass fill:#64748b,stroke:#334155,color:#fff
    classDef externalClass fill:#f59e0b,stroke:#d97706,color:#fff

    class UI1,UI2,UI3,UI4 interfaceClass
    class HR,NQ,NL coreClass
    class QDB,NDB,Files storageClass
    class Mistral externalClass
```

## Classification des Questions

```mermaid
flowchart LR
    Q[Question]

    subgraph Classification["🧭 Analyse"]
        KW1[Keywords Multi-hop:<br/>quels événements, où,<br/>liste, total, somme]
        KW2[Keywords Simple:<br/>qu'est-ce que, décris,<br/>caractéristiques, prix]
        Score{Score<br/>Comparaison}
    end

    Simple[📄 RAG Simple<br/>Qdrant uniquement]
    MultiHop[🔗 RAG Hybride<br/>Qdrant + Neo4j]

    Q --> KW1
    Q --> KW2
    KW1 --> Score
    KW2 --> Score

    Score -->|multi_hop > simple| MultiHop
    Score -->|simple ≥ multi_hop| Simple

    classDef questionClass fill:#9333ea,stroke:#6b21a8,color:#fff
    classDef simpleClass fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef multiClass fill:#10b981,stroke:#059669,color:#fff

    class Q questionClass
    class Simple simpleClass
    class MultiHop multiClass
```

## Requêtes Neo4j Multi-hop

```mermaid
flowchart TB
    Question[Question Utilisateur]

    subgraph Neo4jQuerier["🕸️ Neo4jQuerier.get_graph_context_for_question"]
        Analyze[Analyse de la question]

        Q1[query_events_with_products_sold_at_tradeshows]
        Q2[query_total_co2_saved_by_product]
        Q3[query_tradeshows_sales_by_customer_type]
        Q4[query_rd_projects_for_festival_products]
        Q5[query_products_by_battery_type]
        Q6[query_product_sales_across_tradeshows]
    end

    subgraph Graph["🔗 Neo4j Graph"]
        Product[Product]
        Event[Event]
        TradeShow[TradeShow]
        Sale[Sale]
        RDProject[R&D Project]
        Battery[BatteryType]

        Product -->|DEPLOYED_AT| Event
        Product -->|INCLUDES_PRODUCT| Sale
        Sale -->|SOLD_AT| TradeShow
        Product -->|TARGETS_PRODUCT| RDProject
        Product -->|USES_BATTERY| Battery
    end

    Context[Contexte Graphique<br/>Relations + Agrégations]

    Question --> Analyze
    Analyze -.->|événements + vendus| Q1
    Analyze -.->|co2 + produit| Q2
    Analyze -.->|collectivités| Q3
    Analyze -.->|r&d + festival| Q4
    Analyze -.->|batterie| Q5
    Analyze -.->|salons + produit| Q6

    Q1 --> Graph
    Q2 --> Graph
    Q3 --> Graph
    Q4 --> Graph
    Q5 --> Graph
    Q6 --> Graph

    Graph --> Context

    classDef queryClass fill:#10b981,stroke:#059669,color:#fff
    classDef nodeClass fill:#64748b,stroke:#334155,color:#fff
    classDef contextClass fill:#f59e0b,stroke:#d97706,color:#fff

    class Q1,Q2,Q3,Q4,Q5,Q6 queryClass
    class Product,Event,TradeShow,Sale,RDProject,Battery nodeClass
    class Context contextClass
```

## Pipeline de Génération de Réponse

```mermaid
sequenceDiagram
    participant U as 👤 Utilisateur
    participant A as 🌿 App
    participant R as 🧭 Routeur
    participant Q as 🗄️ Qdrant
    participant N as 🔗 Neo4j
    participant L as 🤖 LLM

    U->>A: Question
    A->>R: classify_question()
    R-->>A: Strategy (simple/multi_hop)

    alt RAG Simple
        A->>Q: Recherche vectorielle (k=3)
        Q-->>A: Documents similaires
        A->>L: Prompt + Contexte vectoriel
        L-->>A: Réponse
    else RAG Hybride
        par Recherche parallèle
            A->>Q: Recherche vectorielle (k=3)
            A->>N: get_graph_context_for_question()
        end
        Q-->>A: Documents similaires
        N-->>A: Relations + Agrégations
        A->>L: Prompt + Contexte vectoriel + Contexte graphique
        L-->>A: Réponse enrichie
    end

    A-->>U: Réponse + Sources
```

## Schéma du Graphe Neo4j

```mermaid
erDiagram
    PRODUCT ||--o{ EVENT : DEPLOYED_AT
    PRODUCT ||--o{ SALE : INCLUDES_PRODUCT
    SALE }o--|| TRADESHOW : SOLD_AT
    RDPROJECT }o--|| PRODUCT : TARGETS_PRODUCT
    PRODUCT }o--|| BATTERYTYPE : USES_BATTERY

    PRODUCT {
        string product_id PK
        string name
        string category
        float battery_capacity
        float avg_selling_price
        float total_cost
    }

    EVENT {
        string name
        string type
        string location
        int attendees
        string co2_reduction
    }

    TRADESHOW {
        string name
        string location
        string date
        float total_sales
        int leads_generated
    }

    SALE {
        string customer_type
        int units
        float total_revenue
    }

    RDPROJECT {
        string name
        string objective
        string status
        string projected_savings
    }

    BATTERYTYPE {
        string type
    }
```

## Flux de Données

```mermaid
flowchart LR
    subgraph Sources["📂 Sources"]
        JSON[JSON Files]
        CSV[CSV Files]
        PDF[PDF Files]
        TXT[TXT Files]
    end

    subgraph Processing["⚙️ Traitement"]
        Load[Document Loaders]
        Split[Text Splitter<br/>chunk_size=1000<br/>overlap=200]
        Embed[Mistral Embeddings<br/>mistral-embed]
        Parse[JSON Parser<br/>Neo4jLoader]
    end

    subgraph Storage["💾 Stockage"]
        QC[(Qdrant Collection<br/>documents_rag)]
        NG[(Neo4j Graph<br/>Nodes + Relations)]
    end

    JSON --> Load
    CSV --> Load
    PDF --> Load
    TXT --> Load

    JSON --> Parse

    Load --> Split
    Split --> Embed
    Embed --> QC

    Parse --> NG

    classDef sourceClass fill:#3b82f6,stroke:#1e40af,color:#fff
    classDef processClass fill:#10b981,stroke:#059669,color:#fff
    classDef storageClass fill:#64748b,stroke:#334155,color:#fff

    class JSON,CSV,PDF,TXT sourceClass
    class Load,Split,Embed,Parse processClass
    class QC,NG storageClass
```

## Métriques et Dashboard

```mermaid
flowchart TB
    subgraph Metrics["📊 Métriques Collectées"]
        QM[Qdrant Metrics<br/>- Nb collections<br/>- Nb vecteurs<br/>- Points indexés]
        NM[Neo4j Metrics<br/>- Nb nœuds<br/>- Nb relations<br/>- Types d'entités]
        PM[Performance<br/>- Temps réponse<br/>- Stratégie utilisée<br/>- Sources consultées]
    end

    subgraph Visualization["📈 Visualisation"]
        Charts[Graphiques Streamlit<br/>- Bar charts<br/>- Metrics cards<br/>- Tables]
    end

    Qdrant[(Qdrant)]
    Neo4j[(Neo4j)]

    Qdrant --> QM
    Neo4j --> NM
    QM --> Charts
    NM --> Charts
    PM --> Charts

    classDef dbClass fill:#64748b,stroke:#334155,color:#fff
    classDef metricClass fill:#f59e0b,stroke:#d97706,color:#fff
    classDef vizClass fill:#ec4899,stroke:#be185d,color:#fff

    class Qdrant,Neo4j dbClass
    class QM,NM,PM metricClass
    class Charts vizClass
```

## Légende

- 📄 **RAG Simple**: Questions descriptives simples (caractéristiques, prix, descriptions)
- 🔗 **RAG Hybride**: Questions complexes nécessitant des relations et agrégations
- 🧭 **Routeur**: Classification automatique basée sur les mots-clés
- 🗄️ **Qdrant**: Base vectorielle pour similarité sémantique
- 🔗 **Neo4j**: Graphe de connaissances pour multi-hop reasoning
- 🤖 **Mistral**: LLM pour génération de réponses et embeddings
