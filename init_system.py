"""
Script d'initialisation du système RAG hybride
Lance tous les tests et vérifie que tout fonctionne
"""

import sys
import os
from pathlib import Path

def check_env():
    """Vérifie les variables d'environnement"""
    print("\n" + "="*70)
    print("🔍 VÉRIFICATION DES VARIABLES D'ENVIRONNEMENT")
    print("="*70)

    from dotenv import load_dotenv
    load_dotenv()

    required_vars = [
        "MISTRAL_API_KEY",
        "QDRANT_ENDPOINT",
        "QDRANT_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE"
    ]

    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Masquer les secrets
            if "KEY" in var or "PASSWORD" in var:
                display = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display = value
            print(f"✅ {var}: {display}")
        else:
            print(f"❌ {var}: MANQUANT")
            all_good = False

    return all_good

def check_neo4j_connection():
    """Teste la connexion Neo4j"""
    print("\n" + "="*70)
    print("🔗 TEST DE CONNEXION NEO4J")
    print("="*70)

    try:
        from neo4j_loader import Neo4jLoader
        loader = Neo4jLoader()

        # Test de connexion simple
        with loader.driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_val = result.single()["test"]
            if test_val == 1:
                print("✅ Connexion Neo4j réussie")
                loader.close()
                return True
    except Exception as e:
        print(f"❌ Erreur de connexion Neo4j: {e}")
        return False

    return False

def load_neo4j_data():
    """Charge les données dans Neo4j"""
    print("\n" + "="*70)
    print("📊 CHARGEMENT DES DONNÉES NEO4J")
    print("="*70)

    try:
        from neo4j_loader import Neo4jLoader
        loader = Neo4jLoader()

        # Vérifier si des données existent déjà
        with loader.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]

            if count > 0:
                print(f"ℹ️  {count} nœuds déjà présents dans Neo4j")
                response = input("Voulez-vous recharger les données? (o/N): ").lower()
                if response != 'o':
                    print("✅ Utilisation des données existantes")
                    loader.close()
                    return True

        # Charger les données
        print("Chargement en cours...")
        loader.load_all()
        loader.verify_data()
        loader.close()
        print("✅ Données Neo4j chargées avec succès")
        return True

    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return False

def check_qdrant_connection():
    """Teste la connexion Qdrant"""
    print("\n" + "="*70)
    print("🔗 TEST DE CONNEXION QDRANT")
    print("="*70)

    try:
        from qdrant_client import QdrantClient
        import os

        client = QdrantClient(
            url=os.getenv("QDRANT_ENDPOINT"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # Test simple
        collections = client.get_collections()
        print(f"✅ Connexion Qdrant réussie")
        print(f"   Collections disponibles: {len(collections.collections)}")

        return True

    except Exception as e:
        print(f"❌ Erreur de connexion Qdrant: {e}")
        return False

def show_summary():
    """Affiche le résumé final"""
    print("\n" + "="*70)
    print("📋 RÉSUMÉ DE L'INITIALISATION")
    print("="*70)
    print("""
Le système RAG hybride est prêt!

🚀 Pour lancer l'application:
   streamlit run app_hybrid.py

📖 Documentation:
   - Consultez README.md pour plus d'informations

🔧 Commandes utiles:
   - Recharger Neo4j: python neo4j_loader.py
   - Requêtes Neo4j: python neo4j_query.py

💡 Conseils:
   - Uploadez vos documents PDF dans l'application
   - Pixtral Vision est activé par défaut pour l'analyse intelligente des PDFs
   - Onglet "Routeur Intelligent" recommandé (choix automatique)
   - Si Neo4j est vide, l'application utilisera uniquement Qdrant pour le RAG
    """)

def main():
    print("="*70)
    print("🎯 INITIALISATION DU SYSTÈME RAG HYBRIDE GREENPOWER")
    print("="*70)

    # Checklist
    checks = [
        ("Variables d'environnement", check_env),
        ("Connexion Neo4j", check_neo4j_connection),
        ("Connexion Qdrant", check_qdrant_connection),
    ]

    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
            print(f"\n❌ {name} a échoué")
            break

    if not all_passed:
        print("\n❌ L'initialisation a échoué. Vérifiez les erreurs ci-dessus.")
        sys.exit(1)

    # Charger Neo4j (non-bloquant)
    neo4j_loaded = load_neo4j_data()
    if not neo4j_loaded:
        print("\n⚠️  Le chargement Neo4j a échoué, mais l'application peut fonctionner avec Qdrant uniquement")

    # Résumé
    show_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Initialisation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
