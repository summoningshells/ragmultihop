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
from dashboard import render_dashboard
from pixtral_processor import PixtralPDFProcessor

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
    """Version classique pypdf (fallback)"""
    reader = PdfReader(file_path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            documents.append(
                Document(page_content=text, metadata={"source": file_path, "type": "pdf", "page": i})
            )
    return documents

def load_pdf_with_pixtral(file_path):
    """
    Charge un PDF avec traitement Pixtral optionnel.
    Fallback gracieux vers pypdf en cas d'erreur.
    """
    # Récupérer le paramètre use_pixtral depuis session_state
    use_pixtral = st.session_state.get('use_pixtral', True)

    if not use_pixtral:
        return load_pdf(file_path)

    try:
        processor = PixtralPDFProcessor(
            mistral_api_key=MISTRAL_API_KEY,
            model="pixtral-12b-2409",
            cache_images=False
        )

        # Callback de progression pour Streamlit
        def progress_callback(current, total):
            st.sidebar.info(f"🔍 Analyse Pixtral: Page {current}/{total}")

        documents = processor.process_pdf_complete(
            file_path,
            dpi=200,
            progress_callback=progress_callback
        )

        if documents:
            st.sidebar.success(f"✅ {len(documents)} chunks enrichis créés avec Pixtral!")

        return documents

    except Exception as e:
        st.warning(f"⚠️ Erreur Pixtral pour {Path(file_path).name}, fallback sur pypdf: {e}")
        return load_pdf(file_path)

def load_documents_from_directory(directory):
    all_documents = []
    data_path = Path(directory)

    if not data_path.exists():
        return all_documents

    loaders = {
        '.txt': load_txt,
        '.json': load_json,
        '.csv': load_csv,
        '.pdf': load_pdf_with_pixtral  # Utilise Pixtral par défaut
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
    st.set_page_config(
        page_title="RAG Hybride GreenPower",
        layout="wide",
        page_icon="🌿",
        initial_sidebar_state="expanded"
    )

    # CSS personnalisé pour améliorer l'UI
    st.markdown("""
        <style>
        /* Amélioration des titres */
        h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(120deg, #10b981, #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            padding-bottom: 1rem;
        }

        h2 {
            font-weight: 600;
            padding-top: 1rem;
            border-bottom: 2px solid #10b981;
            padding-bottom: 0.5rem;
        }

        h3 {
            font-weight: 600;
            color: #10b981;
            margin-top: 1.5rem;
        }

        /* Amélioration des cartes métriques */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #10b981;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Amélioration des expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            background-color: rgba(16, 185, 129, 0.1);
            border-radius: 8px;
            padding: 0.5rem;
        }

        /* Amélioration des inputs */
        .stTextInput input {
            border-radius: 8px;
            border: 2px solid #10b981;
            padding: 0.75rem;
            font-size: 1rem;
        }

        .stTextInput input:focus {
            border-color: #059669;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        }

        /* Amélioration des boutons */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }

        /* Amélioration des tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(16, 185, 129, 0.05);
            padding: 0.5rem;
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: transparent;
            border-radius: 8px;
            font-weight: 600;
            padding: 0 1.5rem;
        }

        .stTabs [aria-selected="true"] {
            background-color: #10b981;
        }

        /* Amélioration des messages info/success */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid #10b981;
            padding: 1rem;
        }

        /* Amélioration du spinner */
        .stSpinner > div {
            border-top-color: #10b981;
        }

        /* Amélioration de la sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(16, 185, 129, 0.05) 0%, transparent 100%);
        }

        /* Amélioration des dividers */
        hr {
            margin: 2rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #10b981, transparent);
        }

        /* Amélioration des dataframes */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }

        /* Animation de fade-in pour le contenu */
        .element-container {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Style des badges personnalisés */
        .badge {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            font-size: 0.85rem;
            font-weight: 600;
            line-height: 1;
            border-radius: 6px;
            margin: 0.25rem;
        }

        .badge-success {
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
        }

        .badge-info {
            background-color: rgba(59, 130, 246, 0.2);
            color: #3b82f6;
            border: 1px solid #3b82f6;
        }
        </style>
    """, unsafe_allow_html=True)

    # En-tête avec style amélioré
    st.title("🌿 RAG Hybride - GreenPower Solutions")
    st.markdown("""
        <p style='font-size: 1.2rem; color: #64748b; margin-bottom: 2rem;'>
            Système RAG intelligent combinant recherche vectorielle et raisonnement graphique
        </p>
    """, unsafe_allow_html=True)

    # Initialisation
    qdrant_client, embeddings, llm, hybrid_rag = init_components()

    # Sidebar améliorée
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;'>
                📂 Documents chargés depuis <code>data/</code>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📤 Importer des documents")
        st.markdown("""
            <div style='background: rgba(59, 130, 246, 0.1); padding: 0.5rem; border-radius: 6px; margin-bottom: 0.75rem; font-size: 0.85rem;'>
                💡 Glissez-déposez vos fichiers ci-dessous
            </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Choisissez des fichiers",
            type=['txt', 'json', 'csv', 'pdf'],
            accept_multiple_files=True,
            help="Formats supportés: TXT, JSON, CSV, PDF",
            label_visibility="collapsed"
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} fichier(s) sélectionné(s)**")

            if st.button("📥 Sauvegarder et indexer", use_container_width=True, type="primary"):
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)

                success_count = 0
                error_count = 0

                with st.spinner("💾 Sauvegarde en cours..."):
                    for uploaded_file in uploaded_files:
                        try:
                            file_path = data_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            success_count += 1
                        except Exception as e:
                            st.error(f"❌ Erreur pour {uploaded_file.name}: {e}")
                            error_count += 1

                if success_count > 0:
                    st.success(f"✅ {success_count} fichier(s) sauvegardé(s)!")
                    st.info("🔄 Rechargez la page pour indexer les nouveaux documents")

                    # Bouton pour recharger immédiatement
                    if st.button("🔄 Recharger maintenant", use_container_width=True):
                        st.cache_resource.clear()
                        st.rerun()

                if error_count > 0:
                    st.warning(f"⚠️ {error_count} erreur(s) rencontrée(s)")

        st.divider()

        st.markdown("### 🔧 Actions système")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Recharger", use_container_width=True, help="Recharge tous les composants"):
                st.cache_resource.clear()
                st.rerun()

        with col2:
            if st.button("🗑️ Reset", use_container_width=True, help="Réinitialise la collection Qdrant"):
                # Supprimer la collection Qdrant
                try:
                    qdrant_client.delete_collection(COLLECTION_NAME)
                    st.cache_resource.clear()
                    st.success("✅ Collection supprimée")
                    st.rerun()
                except Exception as e:
                    st.warning(f"⚠️ Erreur: {e}")

        st.divider()

        st.markdown("### 🔗 Graphe Neo4j")
        if st.button("📊 Charger Neo4j", use_container_width=True, help="Charge les données dans Neo4j"):
            with st.spinner("⏳ Chargement du graphe..."):
                from neo4j_loader import Neo4jLoader
                loader = Neo4jLoader()
                try:
                    loader.load_all()
                    st.success("✅ Graphe chargé!")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                finally:
                    loader.close()

        st.divider()

        st.markdown("### 🎨 Mode Vision Pixtral")
        use_pixtral = st.toggle(
            "Activer Pixtral Vision pour PDFs",
            value=True,
            help="Utilise Pixtral pour analyser visuellement les PDFs (tableaux, images, structure)",
            key="use_pixtral"
        )

        if use_pixtral:
            st.markdown("""
                <div style='background: rgba(147, 51, 234, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 4px solid #9333ea; margin: 0.5rem 0; font-size: 0.85rem;'>
                    ✨ <strong>Mode Vision activé</strong><br>
                    Extraction intelligente de texte, tableaux et images/graphiques
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: rgba(100, 116, 139, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.85rem;'>
                    Mode extraction texte classique (pypdf)
                </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 💡 Exemples de questions")

        with st.expander("📄 Questions simples", expanded=False):
            st.markdown("*Pour RAG classique (Qdrant)*")
            for i, q in enumerate(QueryExamples.SIMPLE_QUESTIONS[:3], 1):
                st.markdown(f"""
                    <div style='background: rgba(59, 130, 246, 0.1); padding: 0.5rem; border-radius: 6px; margin: 0.3rem 0; font-size: 0.85rem;'>
                        <strong>{i}.</strong> {q}
                    </div>
                """, unsafe_allow_html=True)

        with st.expander("🔗 Questions multi-hop", expanded=False):
            st.markdown("*Pour RAG+Graph (Qdrant+Neo4j)*")
            for i, q in enumerate(QueryExamples.MULTI_HOP_QUESTIONS[:4], 1):
                st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.1); padding: 0.5rem; border-radius: 6px; margin: 0.3rem 0; font-size: 0.85rem;'>
                        <strong>{i}.</strong> {q}
                    </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Footer avec info
        st.markdown("""
            <div style='background: rgba(100, 116, 139, 0.1); padding: 0.75rem; border-radius: 8px; text-align: center; font-size: 0.85rem; margin-top: 1rem;'>
                🌿 <strong>GreenPower Solutions</strong><br>
                Système RAG Hybride v1.0
            </div>
        """, unsafe_allow_html=True)

    # Chargement et indexation
    with st.spinner("⏳ Chargement et indexation des documents..."):
        vector_store, num_chunks = load_and_index_documents(qdrant_client, embeddings)

    if vector_store is None:
        st.markdown("""
            <div style='background: rgba(251, 191, 36, 0.2); padding: 1rem; border-radius: 8px; border-left: 4px solid #fbbf24; margin: 1rem 0;'>
                ⚠️ <strong>Aucun document trouvé</strong><br>
                Veuillez ajouter des documents dans le dossier <code>data/</code>
            </div>
        """, unsafe_allow_html=True)
        st.info("📝 Formats supportés: .txt, .json, .csv, .pdf")
        return

    st.markdown(f"""
        <div style='background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin: 1rem 0;'>
            ✅ <strong>{num_chunks} chunks</strong> indexés avec succès dans Qdrant
        </div>
    """, unsafe_allow_html=True)

    # Tabs pour les modes RAG et le dashboard
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 RAG Classique (Qdrant)",
        "🔗 RAG+Graph Multi-Hop",
        "🧭 Routeur Intelligent (Auto)",
        "📊 Dashboard Métriques"
    ])

    # TAB 1: RAG Classique
    with tab1:
        st.markdown("### 📄 RAG Classique - Recherche Vectorielle")
        st.markdown("""
            <div style='background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 1.5rem;'>
                <strong>💡 Mode:</strong> Recherche par similarité sémantique dans Qdrant<br>
                <strong>🎯 Idéal pour:</strong> Questions descriptives simples sur les produits, événements ou spécifications
            </div>
        """, unsafe_allow_html=True)

        question_classic = st.text_input(
            "💬 Posez votre question:",
            placeholder="Ex: Qu'est-ce que le produit GreenPower Max?",
            key="classic",
            help="Entrez une question descriptive sur les produits ou événements"
        )

        if question_classic:
            with st.spinner("🔍 Recherche de la réponse..."):
                result = hybrid_rag.query_simple(question_classic, vector_store)

            st.markdown("---")
            st.markdown("### ✨ Réponse")
            st.markdown(f"""
                <div style='background: rgba(16, 185, 129, 0.05); padding: 1.5rem; border-radius: 8px; margin: 1rem 0; font-size: 1.05rem; line-height: 1.6;'>
                    {result["answer"]}
                </div>
            """, unsafe_allow_html=True)

            with st.expander("📚 Sources utilisées (Qdrant)", expanded=False):
                st.caption(f"**{len(result['sources']['vector_docs'])}** documents pertinents trouvés")
                for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                    st.markdown(f"""
                        <div style='background: rgba(100, 116, 139, 0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                            <strong>📄 Source {i}:</strong> <code>{doc.metadata.get('source', 'N/A')}</code>
                        </div>
                    """, unsafe_allow_html=True)
                    st.text(doc.page_content[:300] + "...")
                    if i < len(result["sources"]["vector_docs"]):
                        st.divider()

    # TAB 2: RAG+Graph
    with tab2:
        st.markdown("### 🔗 RAG+Graph Multi-Hop - Raisonnement Relationnel")
        st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin-bottom: 1.5rem;'>
                <strong>💡 Mode:</strong> Combine Qdrant (documents) + Neo4j (relations graphiques)<br>
                <strong>🎯 Idéal pour:</strong> Questions complexes nécessitant des agrégations, relations multi-étapes ou raisonnement sur le graphe
            </div>
        """, unsafe_allow_html=True)

        question_graph = st.text_input(
            "💬 Posez votre question:",
            placeholder="Ex: Quels événements ont utilisé des produits vendus à Pollutec Paris?",
            key="graph",
            help="Entrez une question complexe avec des relations entre entités"
        )

        if question_graph:
            with st.spinner("🔄 Recherche multi-hop en cours..."):
                result = hybrid_rag.query_hybrid(question_graph, vector_store)

            st.markdown("---")
            st.markdown("### ✨ Réponse")
            st.markdown(f"""
                <div style='background: rgba(16, 185, 129, 0.05); padding: 1.5rem; border-radius: 8px; margin: 1rem 0; font-size: 1.05rem; line-height: 1.6;'>
                    {result["answer"]}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Sources de données")
            col1, col2 = st.columns(2)

            with col1:
                with st.expander("📄 Documents Vectoriels (Qdrant)", expanded=False):
                    st.caption(f"**{len(result['sources']['vector_docs'])}** documents consultés")
                    for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                        st.markdown(f"""
                            <div style='background: rgba(59, 130, 246, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;'>
                                <strong>📄 Document {i}:</strong> <code style='font-size: 0.85rem;'>{doc.metadata.get('source', 'N/A')}</code>
                            </div>
                        """, unsafe_allow_html=True)
                        st.text(doc.page_content[:200] + "...")
                        if i < len(result["sources"]["vector_docs"]):
                            st.divider()

            with col2:
                with st.expander("🔗 Relations Graphiques (Neo4j)", expanded=False):
                    graph_ctx = result["sources"]["graph_context"]
                    if graph_ctx:
                        st.caption(f"**{len(graph_ctx)}** requêtes graphiques exécutées")
                        for idx, item in enumerate(graph_ctx, 1):
                            st.markdown(f"""
                                <div style='background: rgba(16, 185, 129, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;'>
                                    <strong>🔍 Requête {idx}:</strong> {item['query_type']}
                                </div>
                            """, unsafe_allow_html=True)
                            st.json(item["results"][:2])  # Afficher 2 premiers résultats
                            if idx < len(graph_ctx):
                                st.divider()
                    else:
                        st.info("ℹ️ Aucune donnée relationnelle trouvée")

    # TAB 3: Routeur intelligent
    with tab3:
        st.markdown("### 🧭 Routeur Intelligent - Mode Automatique")
        st.markdown("""
            <div style='background: rgba(147, 51, 234, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #9333ea; margin-bottom: 1.5rem;'>
                <strong>💡 Mode:</strong> Sélection automatique de la stratégie optimale<br>
                <strong>🎯 Idéal pour:</strong> Laisser l'IA décider entre RAG classique et RAG+Graph selon la complexité de la question
            </div>
        """, unsafe_allow_html=True)

        question_auto = st.text_input(
            "💬 Posez votre question:",
            placeholder="Posez n'importe quelle question...",
            key="auto",
            help="Le système analysera votre question et choisira automatiquement la meilleure stratégie"
        )

        if question_auto:
            # Afficher la décision du routeur
            with st.expander("🧭 Analyse du routeur intelligent", expanded=True):
                routing = hybrid_rag.explain_routing(question_auto)
                st.markdown(f"""
                    <div style='background: rgba(147, 51, 234, 0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                        <strong>🎯 Stratégie sélectionnée:</strong> <span class='badge badge-info' style='font-size: 1rem;'>{routing['strategy']}</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**📝 Explication:** {routing['explanation']}")

            # Exécuter la requête
            with st.spinner("🤖 Traitement intelligent de la question..."):
                result = hybrid_rag.query(question_auto, vector_store)

            st.markdown("---")



            st.markdown("### ✨ Réponse")
            st.markdown(f"""
                <div style='background: rgba(16, 185, 129, 0.05); padding: 1.5rem; border-radius: 8px; margin: 1rem 0; font-size: 1.05rem; line-height: 1.6;'>
                    {result["answer"]}
                </div>
            """, unsafe_allow_html=True)

            # Sources adaptées à la stratégie
            st.markdown("### 📊 Sources consultées")
            if result["strategy"] == "multi_hop":
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("📄 Documents Qdrant", expanded=False):
                        st.caption(f"**{len(result['sources']['vector_docs'])}** documents")
                        for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                            st.markdown(f"""
                                <div style='background: rgba(59, 130, 246, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;'>
                                    <strong>📄 {i}:</strong> <code style='font-size: 0.85rem;'>{doc.metadata.get('source', 'N/A')}</code>
                                </div>
                            """, unsafe_allow_html=True)
                            st.text(doc.page_content[:150] + "...")
                with col2:
                    with st.expander("🔗 Relations Neo4j", expanded=False):
                        graph_ctx = result["sources"].get("graph_context", [])
                        if graph_ctx:
                            st.caption(f"**{len(graph_ctx)}** requêtes graphiques")
                            for idx, item in enumerate(graph_ctx, 1):
                                st.markdown(f"""
                                    <div style='background: rgba(16, 185, 129, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;'>
                                        <strong>🔍 {idx}:</strong> {item['query_type']}
                                    </div>
                                """, unsafe_allow_html=True)
                                st.json(item["results"][:1])
                        else:
                            st.info("ℹ️ Pas de données graphiques utilisées")
            else:
                with st.expander("📄 Documents Qdrant consultés", expanded=False):
                    st.caption(f"**{len(result['sources']['vector_docs'])}** documents pertinents")
                    for i, doc in enumerate(result["sources"]["vector_docs"], 1):
                        st.markdown(f"""
                            <div style='background: rgba(59, 130, 246, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;'>
                                <strong>📄 Source {i}:</strong> <code>{doc.metadata.get('source', 'N/A')}</code>
                            </div>
                        """, unsafe_allow_html=True)
                        st.text(doc.page_content[:300] + "...")
                        if i < len(result["sources"]["vector_docs"]):
                            st.divider()

    # TAB 4: Dashboard Métriques
    with tab4:
        render_dashboard(qdrant_client, hybrid_rag.neo4j_querier, vector_store)

if __name__ == "__main__":
    main()
