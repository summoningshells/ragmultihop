"""
Dashboard de métriques pour le système RAG Hybride
Affiche les statistiques de performance de Qdrant et Neo4j
"""

import time
import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime


class DashboardMetrics:
    """Collecte et affiche les métriques de performance du système"""

    def __init__(self, qdrant_client, neo4j_querier, vector_store):
        self.qdrant_client = qdrant_client
        self.neo4j_querier = neo4j_querier
        self.vector_store = vector_store

    def get_qdrant_metrics(self) -> Dict:
        """Récupère les métriques de Qdrant"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_name = "documents_rag"

            # Vérifier si la collection existe
            collection_exists = any(c.name == collection_name for c in collections.collections)

            if not collection_exists:
                return {
                    "total_collections": len(collections.collections),
                    "documents_count": 0,
                    "vectors_count": 0,
                    "collection_exists": False
                }

            # Récupérer les infos de la collection
            collection_info = self.qdrant_client.get_collection(collection_name)

            # Gérer le cas où vectors est un dict ou un objet direct
            vector_size = 0
            if hasattr(collection_info.config.params, 'vectors'):
                vectors = collection_info.config.params.vectors
                if isinstance(vectors, dict):
                    # Cas multi-vecteurs: prendre la taille du premier vecteur
                    vector_size = next(iter(vectors.values())).size if vectors else 0
                elif hasattr(vectors, 'size'):
                    # Cas vecteur unique
                    vector_size = vectors.size

            return {
                "total_collections": len(collections.collections),
                "documents_count": collection_info.points_count,
                "vectors_count": collection_info.points_count,
                "collection_exists": True,
                "vector_size": vector_size
            }
        except Exception as e:
            st.error(f"Erreur lors de la récupération des métriques Qdrant: {e}")
            return {"error": str(e)}

    def get_neo4j_metrics(self) -> Dict:
        """Récupère les métriques de Neo4j"""
        try:
            with self.neo4j_querier.driver.session() as session:
                # Compter les nœuds par type
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(n) as count
                    ORDER BY count DESC
                """)
                nodes_by_type = {record["label"]: record["count"] for record in result}

                # Compter les relations par type
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                """)
                relations_by_type = {record["rel_type"]: record["count"] for record in result}

                # Total
                total_nodes = sum(nodes_by_type.values())
                total_relations = sum(relations_by_type.values())

                return {
                    "total_nodes": total_nodes,
                    "total_relations": total_relations,
                    "nodes_by_type": nodes_by_type,
                    "relations_by_type": relations_by_type
                }
        except Exception as e:
            st.error(f"Erreur lors de la récupération des métriques Neo4j: {e}")
            return {"error": str(e)}

    def measure_qdrant_search_time(self, question: str, k: int = 3) -> Tuple[float, int]:
        """Mesure le temps de recherche Qdrant"""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

            start_time = time.time()
            results = retriever.invoke(question)
            end_time = time.time()

            return (end_time - start_time) * 1000, len(results)  # en ms
        except Exception as e:
            st.error(f"Erreur lors de la mesure Qdrant: {e}")
            return 0, 0

    def measure_neo4j_query_time(self, query_func, *args) -> Tuple[float, int]:
        """Mesure le temps d'exécution d'une requête Neo4j"""
        try:
            start_time = time.time()
            results = query_func(*args)
            end_time = time.time()

            return (end_time - start_time) * 1000, len(results)  # en ms
        except Exception as e:
            st.error(f"Erreur lors de la mesure Neo4j: {e}")
            return 0, 0


def render_dashboard(qdrant_client, neo4j_querier, vector_store):
    """Affiche le dashboard complet des métriques"""

    st.markdown("### 📊 Dashboard de Performance")
    st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin-bottom: 1.5rem;'>
            📈 Métriques en temps réel du système RAG Hybride - Surveillance Qdrant et Neo4j
        </div>
    """, unsafe_allow_html=True)

    # Initialiser le dashboard
    dashboard = DashboardMetrics(qdrant_client, neo4j_querier, vector_store)

    # Bouton de refresh avec meilleur style
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)

    if auto_refresh:
        st.markdown("""
            <div style='background: rgba(59, 130, 246, 0.15); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #3b82f6; margin: 0.5rem 0;'>
                ⏱️ <strong>Rafraîchissement automatique activé</strong> (toutes les 5 secondes)
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 1: Métriques Qdrant
    st.markdown("### 🔵 Qdrant - Base Vectorielle")
    st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;'>
            Statistiques de la base de données vectorielle
        </div>
    """, unsafe_allow_html=True)

    qdrant_metrics = dashboard.get_qdrant_metrics()

    if "error" not in qdrant_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📚 Collections",
                value=qdrant_metrics.get("total_collections", 0),
                delta=None,
                help="Nombre total de collections Qdrant"
            )

        with col2:
            st.metric(
                label="📄 Documents indexés",
                value=qdrant_metrics.get("documents_count", 0),
                delta=None,
                help="Nombre de documents dans la collection"
            )

        with col3:
            st.metric(
                label="🎯 Vecteurs",
                value=qdrant_metrics.get("vectors_count", 0),
                delta=None,
                help="Nombre total de vecteurs stockés"
            )

        with col4:
            st.metric(
                label="📐 Dimension",
                value=qdrant_metrics.get("vector_size", 0),
                delta=None,
                help="Dimension des vecteurs d'embedding"
            )

        # Test de performance Qdrant
        st.markdown("")
        st.markdown("#### 🚀 Test de performance de recherche")

        test_questions = [
            "Qu'est-ce que GreenPower?",
            "Prix des produits",
            "Événements festivals"
        ]

        if st.button("▶️ Lancer test Qdrant", key="test_qdrant", use_container_width=False):
            with st.spinner("⏳ Test en cours..."):
                perf_data = []

                for question in test_questions:
                    time_ms, results_count = dashboard.measure_qdrant_search_time(question)
                    perf_data.append({
                        "Question": question[:30] + "...",
                        "Temps (ms)": round(time_ms, 2),
                        "Résultats": results_count
                    })

                df = pd.DataFrame(perf_data)

                # Afficher le tableau avec style
                st.markdown("**📋 Résultats des tests:**")
                st.dataframe(df, use_container_width=True)

                # Graphique des temps
                st.markdown("**📊 Temps de réponse par question:**")
                st.bar_chart(df.set_index("Question")["Temps (ms)"])

                # Moyenne
                avg_time = df["Temps (ms)"].mean()
                st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin: 1rem 0;'>
                        ⏱️ <strong>Temps moyen de recherche:</strong> <span style='font-size: 1.2rem; color: #10b981;'>{avg_time:.2f} ms</span>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Métriques Neo4j
    st.markdown("### 🟢 Neo4j - Graphe de Connaissances")
    st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;'>
            Statistiques de la base de données graphique
        </div>
    """, unsafe_allow_html=True)

    neo4j_metrics = dashboard.get_neo4j_metrics()

    if "error" not in neo4j_metrics:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🔘 Nœuds totaux",
                value=neo4j_metrics.get("total_nodes", 0),
                delta=None,
                help="Nombre total de nœuds dans le graphe"
            )

        with col2:
            st.metric(
                label="🔗 Relations totales",
                value=neo4j_metrics.get("total_relations", 0),
                delta=None,
                help="Nombre total de relations entre nœuds"
            )

        # Graphiques de distribution
        st.markdown("")
        st.markdown("#### 📊 Distribution des données")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔘 Nœuds par type**")
            nodes_data = neo4j_metrics.get("nodes_by_type", {})
            if nodes_data:
                df_nodes = pd.DataFrame({
                    "Type": list(nodes_data.keys()),
                    "Nombre": list(nodes_data.values())
                })
                st.bar_chart(df_nodes.set_index("Type"), height=300)
            else:
                st.info("Aucun nœud trouvé")

        with col2:
            st.markdown("**🔗 Relations par type**")
            relations_data = neo4j_metrics.get("relations_by_type", {})
            if relations_data:
                df_relations = pd.DataFrame({
                    "Type": list(relations_data.keys()),
                    "Nombre": list(relations_data.values())
                })
                st.bar_chart(df_relations.set_index("Type"), height=300)
            else:
                st.info("Aucune relation trouvée")

        # Test de performance Neo4j
        st.markdown("")
        st.markdown("#### 🚀 Test de performance des requêtes")

        if st.button("▶️ Lancer test Neo4j", key="test_neo4j", use_container_width=False):
            with st.spinner("⏳ Test en cours..."):
                perf_data = []

                # Test 1: Requête simple
                time_ms, results_count = dashboard.measure_neo4j_query_time(
                    neo4j_querier.query_top_revenue_tradeshows, 5
                )
                perf_data.append({
                    "Requête": "Top salons (simple)",
                    "Temps (ms)": round(time_ms, 2),
                    "Résultats": results_count
                })

                # Test 2: Requête multi-hop
                time_ms, results_count = dashboard.measure_neo4j_query_time(
                    neo4j_querier.query_events_with_products_sold_at_tradeshows
                )
                perf_data.append({
                    "Requête": "Événements multi-hop",
                    "Temps (ms)": round(time_ms, 2),
                    "Résultats": results_count
                })

                # Test 3: Agrégation
                time_ms, results_count = dashboard.measure_neo4j_query_time(
                    neo4j_querier.query_tradeshows_sales_by_customer_type, "collectivites"
                )
                perf_data.append({
                    "Requête": "Agrégation ventes",
                    "Temps (ms)": round(time_ms, 2),
                    "Résultats": results_count
                })

                # Test 4: R&D multi-hop
                time_ms, results_count = dashboard.measure_neo4j_query_time(
                    neo4j_querier.query_rd_projects_for_festival_products
                )
                perf_data.append({
                    "Requête": "R&D festivals (3 sauts)",
                    "Temps (ms)": round(time_ms, 2),
                    "Résultats": results_count
                })

                df = pd.DataFrame(perf_data)

                # Afficher le tableau avec style
                st.markdown("**📋 Résultats des tests:**")
                st.dataframe(df, use_container_width=True)

                # Graphique des temps
                st.markdown("**📊 Temps d'exécution par type de requête:**")
                st.bar_chart(df.set_index("Requête")["Temps (ms)"], height=300)

                # Moyenne
                avg_time = df["Temps (ms)"].mean()
                st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin: 1rem 0;'>
                        ⏱️ <strong>Temps moyen de requête:</strong> <span style='font-size: 1.2rem; color: #10b981;'>{avg_time:.2f} ms</span>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 3: État du système
    st.markdown("### 🔧 État du Système")
    st.markdown("""
        <div style='background: rgba(100, 116, 139, 0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;'>
            Statut opérationnel des composants
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if "error" not in qdrant_metrics:
            st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #10b981;'>
                    <div style='font-size: 2rem;'>🟢</div>
                    <strong>Qdrant</strong><br>
                    <span style='color: #10b981; font-weight: 600;'>Opérationnel</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: rgba(239, 68, 68, 0.15); padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #ef4444;'>
                    <div style='font-size: 2rem;'>🔴</div>
                    <strong>Qdrant</strong><br>
                    <span style='color: #ef4444; font-weight: 600;'>Erreur</span>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        if "error" not in neo4j_metrics:
            st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #10b981;'>
                    <div style='font-size: 2rem;'>🟢</div>
                    <strong>Neo4j</strong><br>
                    <span style='color: #10b981; font-weight: 600;'>Opérationnel</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: rgba(239, 68, 68, 0.15); padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #ef4444;'>
                    <div style='font-size: 2rem;'>🔴</div>
                    <strong>Neo4j</strong><br>
                    <span style='color: #ef4444; font-weight: 600;'>Erreur</span>
                </div>
            """, unsafe_allow_html=True)

    with col3:
        if "error" not in qdrant_metrics and "error" not in neo4j_metrics:
            st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #10b981;'>
                    <div style='font-size: 2rem;'>✅</div>
                    <strong>Système Global</strong><br>
                    <span style='color: #10b981; font-weight: 600;'>Tout fonctionne</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background: rgba(239, 68, 68, 0.15); padding: 1rem; border-radius: 8px; text-align: center; border: 2px solid #ef4444;'>
                    <div style='font-size: 2rem;'>⚠️</div>
                    <strong>Système Global</strong><br>
                    <span style='color: #ef4444; font-weight: 600;'>Problème détecté</span>
                </div>
            """, unsafe_allow_html=True)

    # Timestamp avec style
    st.markdown("")
    st.markdown(f"""
        <div style='background: rgba(100, 116, 139, 0.1); padding: 0.75rem; border-radius: 8px; text-align: center; margin-top: 1rem;'>
            🕒 <strong>Dernière mise à jour:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    """, unsafe_allow_html=True)

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()
