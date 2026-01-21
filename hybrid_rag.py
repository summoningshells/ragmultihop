import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from neo4j_query import Neo4jQuerier

load_dotenv()

class HybridRAG:
    """
    Routeur intelligent qui décide d'utiliser:
    - RAG classique (Qdrant) pour questions simples/descriptives
    - RAG hybride (Qdrant + Neo4j) pour questions relationnelles/multi-hop
    """

    def __init__(self):
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0
        )
        self.neo4j_querier = Neo4jQuerier()

    def close(self):
        self.neo4j_querier.close()

    def classify_question(self, question):
        """
        Classifie la question pour déterminer la stratégie RAG appropriée.
        Retourne: "simple" ou "multi_hop"
        """
        # Keywords pour multi-hop (questions relationnelles/agrégations)
        multi_hop_keywords = [
            # Relations
            "quels événements", "quels salons", "où", "qui",
            "liste", "lister", "tous les", "combien",
            # Agrégations
            "total", "somme", "moyenne", "maximum", "minimum",
            "économisé", "co2", "carbone",
            # Patterns multi-hop
            "vendus à", "utilisés aux", "déployés à", "présentés à",
            "projets r&d", "recherche", "développement",
            # Customer types
            "collectivités", "entreprises", "particuliers",
            # Liens produit-événement
            "festival", "salon", "avec", "par"
        ]

        question_lower = question.lower()

        # Check for multi-hop patterns
        multi_hop_score = sum(1 for keyword in multi_hop_keywords if keyword in question_lower)

        # Questions simples: description, spécifications, prix
        simple_keywords = [
            "qu'est-ce que", "c'est quoi", "décris", "describe",
            "caractéristiques", "specifications", "prix", "coût",
            "comment fonctionne", "fonctionnement",
            "garantie", "warranty", "maintenance"
        ]

        simple_score = sum(1 for keyword in simple_keywords if keyword in question_lower)

        # Decision
        if multi_hop_score > simple_score:
            return "multi_hop"
        else:
            return "simple"

    def query_hybrid(self, question, vector_store):
        """
        RAG Hybride: Combine Qdrant (similarité sémantique) + Neo4j (relations)
        """
        # 1. Récupérer le contexte du graphe Neo4j
        graph_context_raw = self.neo4j_querier.get_graph_context_for_question(question)
        graph_context = self.neo4j_querier.format_graph_context(graph_context_raw)

        # 2. Récupérer le contexte vectoriel de Qdrant
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        vector_docs = retriever.invoke(question)
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs])

        # 3. Créer un prompt enrichi avec les deux contextes
        template = """Tu es un assistant expert sur GreenPower Solutions et leurs produits solaires autonomes.

Tu dois répondre à la question en utilisant DEUX sources de contexte:

1. CONTEXTE VECTORIEL (descriptions détaillées, documents):
{vector_context}

2. CONTEXTE GRAPHE (relations, agrégations, connexions):
{graph_context}

Utilise prioritairement le CONTEXTE GRAPHE pour les informations relationnelles (qui, où, combien, total, etc.)
et le CONTEXTE VECTORIEL pour les descriptions détaillées et spécifications.

Si une information n'est pas présente dans les contextes, dis clairement que tu ne sais pas.
Ne fabrique pas de réponses.

QUESTION: {question}

RÉPONSE:"""

        prompt = ChatPromptTemplate.from_template(template)

        # 4. Générer la réponse
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({
            "question": question,
            "vector_context": vector_context,
            "graph_context": graph_context if graph_context else "Aucune information relationnelle trouvée."
        })

        return {
            "answer": answer,
            "sources": {
                "vector_docs": vector_docs,
                "graph_context": graph_context_raw
            },
            "strategy": "hybrid"
        }

    def query_simple(self, question, vector_store):
        """
        RAG Simple: Utilise seulement Qdrant (similarité vectorielle)
        """
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        vector_docs = retriever.invoke(question)
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs])

        template = """Tu dois répondre UNIQUEMENT à partir des informations fournies dans le CONTEXTE ci-dessous.
Si une information n'est pas présente dans le CONTEXTE, dis clairement que tu ne sais pas.
Ne fabrique pas de réponses.

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({
            "question": question,
            "context": vector_context
        })

        return {
            "answer": answer,
            "sources": {
                "vector_docs": vector_docs
            },
            "strategy": "simple"
        }

    def query(self, question, vector_store, force_strategy=None):
        """
        Point d'entrée principal avec routage intelligent.

        Args:
            question: La question de l'utilisateur
            vector_store: Le vector store Qdrant
            force_strategy: "simple", "multi_hop", ou None (auto)

        Returns:
            dict avec answer, sources, strategy
        """
        # Déterminer la stratégie
        if force_strategy:
            strategy = force_strategy
        else:
            strategy = self.classify_question(question)

        # Router vers la bonne méthode
        if strategy == "multi_hop":
            return self.query_hybrid(question, vector_store)
        else:
            return self.query_simple(question, vector_store)

    def explain_routing(self, question):
        """
        Explique pourquoi une question est routée vers simple ou multi-hop
        """
        strategy = self.classify_question(question)

        if strategy == "multi_hop":
            explanation = """
Cette question nécessite un RAG HYBRIDE (Qdrant + Neo4j) car elle implique:
- Des relations entre entités (produits, événements, salons)
- Des agrégations (total, somme, liste complète)
- Du multi-hop reasoning (suivre des chemins dans le graphe)

Le graphe Neo4j va permettre de:
- Naviguer entre les nœuds liés
- Calculer des agrégations
- Trouver des patterns complexes
"""
        else:
            explanation = """
Cette question peut être traitée avec un RAG SIMPLE (Qdrant uniquement) car elle demande:
- Une description ou spécification
- Des informations contenues dans les documents
- Pas de relations complexes ou agrégations

La recherche vectorielle suffit pour trouver la réponse.
"""

        return {
            "strategy": strategy,
            "explanation": explanation.strip()
        }


class QueryExamples:
    """Exemples de questions pour tester le système"""

    SIMPLE_QUESTIONS = [
        "Qu'est-ce que le produit GreenPower Max?",
        "Quelles sont les caractéristiques du PG-U01?",
        "Quel est le prix du GreenPower Compact?",
        "Comment fonctionne un générateur solaire autonome?",
        "Quelle est la capacité de la batterie du PG-M01?",
        "Quels sont les avantages de la location?",
    ]

    MULTI_HOP_QUESTIONS = [
        "Quels événements ont utilisé des produits vendus à Pollutec Paris?",
        "Quel est le CO2 total économisé par le produit PG-M01?",
        "Quels salons ont généré le plus de revenus avec les collectivités?",
        "Quels projets R&D visent les produits utilisés aux festivals?",
        "Quels produits avec batteries LiFePO4 ont été vendus?",
        "Dans quels salons le PG-U01 a-t-il été vendu?",
    ]
