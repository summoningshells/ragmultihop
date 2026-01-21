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

    st.header("📊 Dashboard de Performance")
    st.markdown("Métriques en temps réel du système RAG Hybride")

    # Initialiser le dashboard
    dashboard = DashboardMetrics(qdrant_client, neo4j_querier, vector_store)

    # Bouton de refresh
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Rafraîchir"):
            st.rerun()
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)

    if auto_refresh:
        st.info("⏱️ Rafraîchissement automatique activé")

    st.divider()

    # Section 1: Métriques Qdrant
    st.subheader("🔵 Qdrant - Base Vectorielle")

    qdrant_metrics = dashboard.get_qdrant_metrics()

    if "error" not in qdrant_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Collections",
                value=qdrant_metrics.get("total_collections", 0)
            )

        with col2:
            st.metric(
                label="Documents indexés",
                value=qdrant_metrics.get("documents_count", 0)
            )

        with col3:
            st.metric(
                label="Vecteurs",
                value=qdrant_metrics.get("vectors_count", 0)
            )

        with col4:
            st.metric(
                label="Dimension vectorielle",
                value=qdrant_metrics.get("vector_size", 0)
            )

        # Test de performance Qdrant
        st.markdown("**Test de performance de recherche**")

        test_questions = [
            "Qu'est-ce que GreenPower?",
            "Prix des produits",
            "Événements festivals"
        ]

        if st.button("▶️ Lancer test Qdrant", key="test_qdrant"):
            with st.spinner("Test en cours..."):
                perf_data = []

                for question in test_questions:
                    time_ms, results_count = dashboard.measure_qdrant_search_time(question)
                    perf_data.append({
                        "Question": question[:30] + "...",
                        "Temps (ms)": round(time_ms, 2),
                        "Résultats": results_count
                    })

                df = pd.DataFrame(perf_data)

                # Afficher le tableau
                st.dataframe(df, use_container_width=True)

                # Graphique des temps
                st.bar_chart(df.set_index("Question")["Temps (ms)"])

                # Moyenne
                avg_time = df["Temps (ms)"].mean()
                st.success(f"⏱️ Temps moyen de recherche: **{avg_time:.2f} ms**")

    st.divider()

    # Section 2: Métriques Neo4j
    st.subheader("🟢 Neo4j - Graphe de Connaissances")

    neo4j_metrics = dashboard.get_neo4j_metrics()

    if "error" not in neo4j_metrics:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Nœuds totaux",
                value=neo4j_metrics.get("total_nodes", 0)
            )

        with col2:
            st.metric(
                label="Relations totales",
                value=neo4j_metrics.get("total_relations", 0)
            )

        # Graphiques de distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Distribution des nœuds par type**")
            nodes_data = neo4j_metrics.get("nodes_by_type", {})
            if nodes_data:
                df_nodes = pd.DataFrame({
                    "Type": list(nodes_data.keys()),
                    "Nombre": list(nodes_data.values())
                })
                st.bar_chart(df_nodes.set_index("Type"))

        with col2:
            st.markdown("**Distribution des relations par type**")
            relations_data = neo4j_metrics.get("relations_by_type", {})
            if relations_data:
                df_relations = pd.DataFrame({
                    "Type": list(relations_data.keys()),
                    "Nombre": list(relations_data.values())
                })
                st.bar_chart(df_relations.set_index("Type"))

        # Test de performance Neo4j
        st.markdown("**Test de performance des requêtes**")

        if st.button("▶️ Lancer test Neo4j", key="test_neo4j"):
            with st.spinner("Test en cours..."):
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

                # Afficher le tableau
                st.dataframe(df, use_container_width=True)

                # Graphique des temps
                st.bar_chart(df.set_index("Requête")["Temps (ms)"])

                # Moyenne
                avg_time = df["Temps (ms)"].mean()
                st.success(f"⏱️ Temps moyen de requête: **{avg_time:.2f} ms**")

    st.divider()

    # Section 3: État du système
    st.subheader("🔧 État du Système")

    col1, col2, col3 = st.columns(3)

    with col1:
        qdrant_status = "🟢 Opérationnel" if "error" not in qdrant_metrics else "🔴 Erreur"
        st.markdown(f"**Qdrant:** {qdrant_status}")

    with col2:
        neo4j_status = "🟢 Opérationnel" if "error" not in neo4j_metrics else "🔴 Erreur"
        st.markdown(f"**Neo4j:** {neo4j_status}")

    with col3:
        overall_status = "🟢 Système OK" if ("error" not in qdrant_metrics and "error" not in neo4j_metrics) else "🔴 Problème détecté"
        st.markdown(f"**Global:** {overall_status}")

    # Timestamp
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()
