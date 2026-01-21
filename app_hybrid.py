import os
import json
import csv
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient

from pypdf import PdfReader
from hybrid_rag import HybridRAG, QueryExamples

# Configuration
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "documents_rag"

# Fonctions de chargement (identiques à app.py)
def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path, "type": "txt"})]

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return [Document(page_content=content, metadata={"source": file_path, "type": "json"})]

def load_csv(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = "\n".join([f"{k}: {v}" for k, v in row.items()])
            documents.append(
                Document(page_content=content, metadata={"source": file_path, "type": "csv", "row": i})
            )
    return documents

def load_pdf(file_path):
    reader = PdfReader(file_path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            documents.append(
                Document(page_content=text, metadata={"source": file_path, "type": "pdf", "page": i})
            )
    return documents

def load_documents_from_directory(directory):
    all_documents = []
    data_path = Path(directory)

    if not data_path.exists():
        return all_documents

    loaders = {
        '.txt': load_txt,
        '.json': load_json,
        '.csv': load_csv,
        '.pdf': load_pdf
    }

    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in loaders:
                try:
                    docs = loaders[ext](str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    st.warning(f"Erreur lors du chargement de {file_path.name}: {e}")

    return all_documents

# Initialisation du cache Streamlit
@st.cache_resource
def init_components():
    qdrant_client = QdrantClient(
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_API_KEY
    )

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0
    )

    hybrid_rag = HybridRAG()

    return qdrant_client, embeddings, llm, hybrid_rag

@st.cache_resource
def load_and_index_documents(_qdrant_client, _embeddings):
    documents = load_documents_from_directory("data")

    if not documents:
        return None, 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Vérifier si la collection existe déjà
    try:
        collections = _qdrant_client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)

        if collection_exists:
            # Collection existe, l'utiliser directement
            vector_store = QdrantVectorStore(
                client=_qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=_embeddings
            )
            return vector_store, len(splits)
    except:
        pass

    # Collection n'existe pas, la créer
    vector_store = QdrantVectorStore.from_documents(
        splits,
        _embeddings,
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )

    return vector_store, len(splits)

# Interface Streamlit
def main():
    st.set_page_config(page_title="RAG Hybride GreenPower", layout="wide")

    st.title("RAG Hybride - GreenPower Solutions")
    st.markdown("Système RAG avec Neo4j pour multi-hop reasoning")

    # Initialisation
    qdrant_client, embeddings, llm, hybrid_rag = init_components()

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info("Documents chargés depuis 'data/'")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Recharger"):
                st.cache_resource.clear()
                st.rerun()

        with col2:
            if st.button("🗑️ Réinitialiser"):
                # Supprimer la collection Qdrant
                try:
                    qdrant_client.delete_collection(COLLECTION_NAME)
                    st.cache_resource.clear()
                    st.success("Collection supprimée")
                    st.rerun()
                except Exception as e:
                    st.warning(f"Erreur: {e}")

        st.divider()

        st.subheader("État du graphe Neo4j")
        if st.button("📊 Charger Neo4j"):
            with st.spinner("Chargement du graphe..."):
                from neo4j_loader import Neo4jLoader
                loader = Neo4jLoader()
                try:
                    loader.load_all()
                    st.success("Graphe Neo4j chargé!")
                except Exception as e:
                    st.error(f"Erreur: {e}")
                finally:
                    loader.close()

        st.divider()

        st.subheader("Exemples de questions")
        st.markdown("**Questions simples (RAG classique):**")
        for q in QueryExamples.SIMPLE_QUESTIONS[:3]:
            st.caption(f"• {q}")

        st.markdown("**Questions multi-hop (RAG+Graph):**")
        for q in QueryExamples.MULTI_HOP_QUESTIONS[:4]:
            st.caption(f"• {q}")

    # Chargement et indexation
    with st.spinner("Chargement et indexation des documents..."):
        vector_store, num_chunks = load_and_index_documents(qdrant_client, embeddings)

    if vector_store is None:
        st.warning("Aucun document trouvé dans le dossier 'data/'.")
        st.info("Formats supportés: .txt, .json, .csv, .pdf")
        return

    st.success(f"{num_chunks} chunks indexés dans Qdrant")

    # Tabs pour les deux modes
    tab1, tab2, tab3 = st.tabs([
        "🤖 RAG Classique (Qdrant)",
        "🔗 RAG+Graph Multi-Hop",
        "🧭 Routeur Intelligent (Auto)"
    ])

    # TAB 1: RAG Classique
    with tab1:
        st.header("RAG Classique - Recherche Vectorielle")
        st.info("Utilise uniquement Qdrant pour la similarité sémantique. Idéal pour questions descriptives.")

        question_classic = st.text_input(
            "Votre question:",
            placeholder="Ex: Qu'est-ce que le produit GreenPower Max?",
            key="classic"
        )

        if question_classic:
            with st.spinner("Recherche de la réponse..."):
                result = hybrid_rag.query_simple(question_classic, vector_store)

            st.markdown("### Réponse")
            st.write(result["answer"])

            with st.expander("Sources utilisées (Qdrant)"):
                for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                    st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

    # TAB 2: RAG+Graph
    with tab2:
        st.header("RAG+Graph Multi-Hop - Raisonnement Relationnel")
        st.info("Combine Qdrant (documents) + Neo4j (relations). Idéal pour questions avec agrégations/relations.")

        question_graph = st.text_input(
            "Votre question:",
            placeholder="Ex: Quels événements ont utilisé des produits vendus à Pollutec Paris?",
            key="graph"
        )

        if question_graph:
            with st.spinner("Recherche multi-hop..."):
                result = hybrid_rag.query_hybrid(question_graph, vector_store)

            st.markdown("### Réponse")
            st.write(result["answer"])

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("📄 Sources Qdrant (Documents)"):
                    for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                        st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
                        st.text(doc.page_content[:200] + "...")
                        st.divider()

            with col2:
                with st.expander("🔗 Sources Neo4j (Graphe)"):
                    graph_ctx = result["sources"]["graph_context"]
                    if graph_ctx:
                        for item in graph_ctx:
                            st.markdown(f"**Query:** {item['query_type']}")
                            st.json(item["results"][:2])  # Afficher 2 premiers résultats
                    else:
                        st.info("Aucune donnée relationnelle trouvée")

    # TAB 3: Routeur intelligent
    with tab3:
        st.header("Routeur Intelligent - Mode Automatique")
        st.info("Le système choisit automatiquement la meilleure stratégie (RAG classique ou RAG+Graph)")

        question_auto = st.text_input(
            "Votre question:",
            placeholder="Posez n'importe quelle question...",
            key="auto"
        )

        if question_auto:
            # Afficher la décision du routeur
            with st.expander("🧭 Décision du routeur", expanded=True):
                routing = hybrid_rag.explain_routing(question_auto)
                st.markdown(f"**Stratégie choisie:** `{routing['strategy']}`")
                st.markdown(routing['explanation'])

            # Exécuter la requête
            with st.spinner("Traitement de la question..."):
                result = hybrid_rag.query(question_auto, vector_store)

            # Afficher badge de stratégie
            if result["strategy"] == "multi_hop":
                st.success("🔗 Stratégie: RAG Hybride (Qdrant + Neo4j)")
            else:
                st.info("📄 Stratégie: RAG Classique (Qdrant uniquement)")

            st.markdown("### Réponse")
            st.write(result["answer"])

            # Sources adaptées à la stratégie
            if result["strategy"] == "multi_hop":
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("📄 Sources Qdrant"):
                        for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                            st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
                            st.text(doc.page_content[:150] + "...")
                with col2:
                    with st.expander("🔗 Sources Neo4j"):
                        graph_ctx = result["sources"].get("graph_context", [])
                        if graph_ctx:
                            for item in graph_ctx:
                                st.markdown(f"**{item['query_type']}**")
                                st.json(item["results"][:1])
            else:
                with st.expander("📄 Sources Qdrant"):
                    for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                        st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'N/A')}")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

if __name__ == "__main__":
    main()
