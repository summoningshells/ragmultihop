import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class Neo4jQuerier:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def query_events_with_products_sold_at_tradeshows(self, location=None):
        """
        Question multi-hop: Quels événements ont utilisé des produits vendus à un salon?
        DEPLOYED_AT <- Product -> INCLUDES_PRODUCT <- Sale -> SOLD_AT -> TradeShow
        """
        with self.driver.session() as session:
            if location:
                query = """
                MATCH (p:Product)-[:DEPLOYED_AT]->(e:Event)
                MATCH (p)<-[inc:INCLUDES_PRODUCT]-(s:Sale)-[:SOLD_AT]->(t:TradeShow)
                WHERE toLower(t.location) CONTAINS toLower($location)
                RETURN DISTINCT
                    e.name as event_name,
                    e.type as event_type,
                    e.location as event_location,
                    collect(DISTINCT p.name) as products_used,
                    collect(DISTINCT t.name) as tradeshows
                ORDER BY e.name
                """
                result = session.run(query, location=location)
            else:
                query = """
                MATCH (p:Product)-[:DEPLOYED_AT]->(e:Event)
                MATCH (p)<-[inc:INCLUDES_PRODUCT]-(s:Sale)-[:SOLD_AT]->(t:TradeShow)
                RETURN DISTINCT
                    e.name as event_name,
                    e.type as event_type,
                    e.location as event_location,
                    collect(DISTINCT p.name) as products_used,
                    collect(DISTINCT t.name) as tradeshows
                ORDER BY e.name
                """
                result = session.run(query)

            return [dict(record) for record in result]

    def query_total_co2_saved_by_product(self, product_id):
        """
        Question multi-hop: Quel est le CO2 total économisé par tous les déploiements d'un produit?
        Product -> DEPLOYED_AT -> Event (avec co2_reduction)
        """
        with self.driver.session() as session:
            query = """
            MATCH (p:Product {product_id: $product_id})-[d:DEPLOYED_AT]->(e:Event)
            WITH p, e, d,
                 CASE
                   WHEN e.co2_reduction CONTAINS 'tonnes' THEN
                     toFloat(split(e.co2_reduction, ' ')[0]) * COALESCE(d.quantity, 1)
                   ELSE 0
                 END as co2_saved
            RETURN p.name as product_name,
                   p.product_id as product_id,
                   sum(co2_saved) as total_co2_saved_tonnes,
                   count(e) as num_deployments,
                   collect(e.name) as events
            """
            result = session.run(query, product_id=product_id)
            return [dict(record) for record in result]

    def query_tradeshows_sales_by_customer_type(self, customer_type="collectivites"):
        """
        Question multi-hop: Liste les salons où on a vendu aux collectivités avec le revenu total
        TradeShow <- SOLD_AT <- Sale (customer_type = collectivites)
        """
        with self.driver.session() as session:
            query = """
            MATCH (s:Sale {customer_type: $customer_type})-[:SOLD_AT]->(t:TradeShow)
            MATCH (s)-[inc:INCLUDES_PRODUCT]->(p:Product)
            RETURN t.name as tradeshow_name,
                   t.location as location,
                   t.date as date,
                   sum(s.total_revenue) as total_revenue,
                   sum(s.units) as total_units,
                   collect(DISTINCT p.name) as products_sold
            ORDER BY total_revenue DESC
            """
            result = session.run(query, customer_type=customer_type)
            return [dict(record) for record in result]

    def query_rd_projects_for_festival_products(self):
        """
        Question multi-hop: Quels projets R&D visent à réduire les coûts des produits utilisés aux festivals?
        Event (type=festival) <- DEPLOYED_AT <- Product <- TARGETS_PRODUCT <- RDProject
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Event)-[:DEPLOYED_AT]-(p:Product)<-[:TARGETS_PRODUCT]-(r:RDProject)
            WHERE toLower(e.type) CONTAINS 'festival' OR toLower(e.name) CONTAINS 'festival'
            WITH r, p, collect(DISTINCT e.name) as festivals
            RETURN DISTINCT r.name as rd_project_name,
                   r.objective as objective,
                   r.status as status,
                   r.projected_savings as projected_savings,
                   collect(DISTINCT p.name) as target_products,
                   festivals
            ORDER BY r.name
            """
            result = session.run(query)
            return [dict(record) for record in result]

    def query_products_by_battery_type(self, battery_type):
        """
        Recherche simple: Quels produits utilisent un type de batterie spécifique?
        """
        with self.driver.session() as session:
            query = """
            MATCH (p:Product)-[:USES_BATTERY]->(b:BatteryType)
            WHERE toLower(b.type) CONTAINS toLower($battery_type)
            RETURN p.product_id as product_id,
                   p.name as product_name,
                   p.battery_capacity as battery_capacity,
                   b.type as battery_type,
                   p.total_cost as total_cost,
                   p.avg_selling_price as price
            ORDER BY p.avg_selling_price
            """
            result = session.run(query, battery_type=battery_type)
            return [dict(record) for record in result]

    def query_top_revenue_tradeshows(self, limit=5):
        """
        Recherche simple: Top salons par revenu
        """
        with self.driver.session() as session:
            query = """
            MATCH (t:TradeShow)
            RETURN t.name as name,
                   t.location as location,
                   t.date as date,
                   t.total_sales as total_sales,
                   t.leads_generated as leads_generated
            ORDER BY t.total_sales DESC
            LIMIT $limit
            """
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]

    def query_product_sales_across_tradeshows(self, product_id):
        """
        Question multi-hop: Dans quels salons un produit a-t-il été vendu et en quelle quantité?
        """
        with self.driver.session() as session:
            query = """
            MATCH (p:Product {product_id: $product_id})<-[inc:INCLUDES_PRODUCT]-(s:Sale)-[:SOLD_AT]->(t:TradeShow)
            RETURN t.name as tradeshow_name,
                   t.location as location,
                   t.date as date,
                   s.customer_type as customer_type,
                   inc.quantity as quantity,
                   s.total_revenue as sale_revenue
            ORDER BY t.date DESC
            """
            result = session.run(query, product_id=product_id)
            return [dict(record) for record in result]

    def query_events_powered_by_product_type(self, category):
        """
        Question: Quels événements ont été alimentés par un type de produit?
        """
        with self.driver.session() as session:
            query = """
            MATCH (p:Product)-[d:DEPLOYED_AT]->(e:Event)
            WHERE toLower(p.category) CONTAINS toLower($category)
            RETURN e.name as event_name,
                   e.type as event_type,
                   e.location as location,
                   e.attendees as attendees,
                   e.co2_reduction as co2_saved,
                   collect(p.name) as products_used,
                   sum(d.quantity) as total_units
            ORDER BY e.name
            """
            result = session.run(query, category=category)
            return [dict(record) for record in result]

    def get_graph_context_for_question(self, question):
        """
        Retourne un contexte du graphe pertinent pour une question donnée.
        Cette fonction analyse la question et exécute les requêtes appropriées.
        """
        question_lower = question.lower()
        context = []

        # Détection de patterns de questions
        if any(word in question_lower for word in ["événements", "events", "déploiements"]):
            if any(word in question_lower for word in ["vendus", "sold", "salon", "tradeshow", "pollutec", "paris"]):
                # Question sur événements avec produits vendus aux salons
                location = None
                if "pollutec" in question_lower or "paris" in question_lower:
                    location = "Paris"
                results = self.query_events_with_products_sold_at_tradeshows(location)
                if results:
                    context.append({
                        "query_type": "events_with_products_sold_at_tradeshows",
                        "results": results
                    })

        if any(word in question_lower for word in ["co2", "carbone", "émissions", "économisé"]):
            # Trouver le produit mentionné
            product_ids = ["PG-U01", "PG-M01", "PG-P01", "PG-C01", "PG-M02"]
            for prod_id in product_ids:
                if prod_id.lower() in question_lower:
                    results = self.query_total_co2_saved_by_product(prod_id)
                    if results:
                        context.append({
                            "query_type": f"total_co2_saved_by_{prod_id}",
                            "results": results
                        })

        if any(word in question_lower for word in ["collectivités", "collectivites", "municipalités"]):
            results = self.query_tradeshows_sales_by_customer_type("collectivites")
            if results:
                context.append({
                    "query_type": "tradeshows_collectivites_sales",
                    "results": results
                })

        if any(word in question_lower for word in ["r&d", "recherche", "développement", "projets"]):
            if any(word in question_lower for word in ["festival", "événements", "coûts", "réduire"]):
                results = self.query_rd_projects_for_festival_products()
                if results:
                    context.append({
                        "query_type": "rd_projects_for_festivals",
                        "results": results
                    })

        if any(word in question_lower for word in ["batterie", "battery", "lifepo4", "tesla"]):
            battery_type = ""
            if "lifepo4" in question_lower:
                battery_type = "LiFePO4"
            elif "tesla" in question_lower:
                battery_type = "Tesla"
            if battery_type:
                results = self.query_products_by_battery_type(battery_type)
                if results:
                    context.append({
                        "query_type": f"products_with_{battery_type}_battery",
                        "results": results
                    })

        if any(word in question_lower for word in ["top", "meilleurs", "plus", "salons", "revenus"]):
            results = self.query_top_revenue_tradeshows(5)
            if results:
                context.append({
                    "query_type": "top_revenue_tradeshows",
                    "results": results
                })

        return context

    def format_graph_context(self, context):
        """
        Formate le contexte du graphe en texte lisible pour le LLM
        """
        if not context:
            return ""

        formatted_parts = []
        for item in context:
            query_type = item["query_type"]
            results = item["results"]

            if not results:
                continue

            formatted_parts.append(f"\n=== Résultats de la requête: {query_type} ===\n")

            for result in results:
                for key, value in result.items():
                    if isinstance(value, list):
                        formatted_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
                    else:
                        formatted_parts.append(f"{key}: {value}")
                formatted_parts.append("---")

        return "\n".join(formatted_parts)

if __name__ == "__main__":
    # Test des requêtes
    querier = Neo4jQuerier()
    try:
        print("\n=== Test: Événements avec produits vendus à Paris ===")
        results = querier.query_events_with_products_sold_at_tradeshows("Paris")
        for r in results[:3]:
            print(r)

        print("\n=== Test: CO2 total économisé par PG-M01 ===")
        results = querier.query_total_co2_saved_by_product("PG-M01")
        for r in results:
            print(r)

        print("\n=== Test: Salons avec ventes aux collectivités ===")
        results = querier.query_tradeshows_sales_by_customer_type("collectivites")
        for r in results[:3]:
            print(r)

        print("\n=== Test: Projets R&D pour festivals ===")
        results = querier.query_rd_projects_for_festival_products()
        for r in results:
            print(r)

    finally:
        querier.close()
