"""
Processeur intelligent de PDFs utilisant Pixtral (Mistral Vision)
pour extraire texte, tableaux, images avec descriptions enrichies.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import base64
import io
import json

from PIL import Image
from pdf2image import convert_from_path
from mistralai import Mistral
from langchain_core.documents import Document


class PixtralPDFProcessor:
    """
    Processeur intelligent de PDFs utilisant Pixtral (Mistral Vision)
    pour extraire texte, tableaux, images avec descriptions enrichies.
    """

    def __init__(
        self,
        mistral_api_key: str,
        model: str = "pixtral-12b-2409",
        cache_images: bool = False,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialise le processeur Pixtral.

        Args:
            mistral_api_key: Clé API Mistral
            model: Modèle Pixtral ("pixtral-12b-2409" ou "pixtral-large-latest")
            cache_images: Si True, sauvegarde les images extraites
            cache_dir: Répertoire de cache (par défaut: data/.pdf_cache)
        """
        self.client = Mistral(api_key=mistral_api_key)
        self.model = model
        self.cache_images = cache_images
        self.cache_dir = cache_dir or Path("data/.pdf_cache")

        if self.cache_images:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def convert_pdf_to_images(
        self,
        pdf_path: str,
        dpi: int = 200
    ) -> List[Image.Image]:
        """
        Convertit chaque page PDF en image PIL.

        Args:
            pdf_path: Chemin du fichier PDF
            dpi: Résolution (300 pour qualité haute, 200 pour performance)

        Returns:
            Liste d'images PIL (une par page)
        """
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt='png'
        )
        return images

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Encode une image PIL en base64 pour l'API Pixtral.

        Args:
            image: Image PIL

        Returns:
            String base64
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_page_with_pixtral(
        self,
        image: Image.Image,
        page_num: int,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyse une page PDF (image) avec Pixtral pour extraire:
        - Texte complet avec structure
        - Tableaux avec descriptions
        - Images/graphiques avec descriptions

        Args:
            image: Image PIL de la page
            page_num: Numéro de page
            custom_prompt: Prompt personnalisé (optionnel)

        Returns:
            Dict avec structured_text, tables, visual_elements, metadata
        """
        base64_image = self.encode_image_to_base64(image)

        # Prompt structuré pour extraction intelligente
        prompt = custom_prompt or """Analyse cette page de document PDF et extrait les informations suivantes au format JSON structuré:

1. **text_content**: Le texte complet de la page avec sa structure (titres, paragraphes, listes)
2. **tables**: Liste de tous les tableaux trouvés avec:
   - description: Description du contenu du tableau
   - headers: En-têtes de colonnes (liste de strings)
   - data_summary: Résumé des données importantes
3. **visual_elements**: Liste de tous les éléments visuels (images, graphiques, diagrammes) avec:
   - type: "image", "chart", "diagram", "logo", etc.
   - description: Description détaillée du contenu
   - position: "top", "middle", "bottom"
4. **document_structure**:
   - has_header: bool
   - has_footer: bool
   - layout_type: "single_column", "multi_column", "mixed"

Réponds UNIQUEMENT en JSON valide, sans markdown."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.0
            )

            # Parse JSON response
            content = response.choices[0].message.content

            # Nettoyer le contenu si markdown est présent
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()

            # Parser le JSON
            analysis = json.loads(content)

            # Valider la structure de base
            if not isinstance(analysis, dict):
                raise ValueError(f"Réponse Pixtral invalide: devrait être un dict, reçu {type(analysis)}")

            # Assurer que les champs requis existent
            if "text_content" not in analysis:
                analysis["text_content"] = ""
            if "tables" not in analysis or not isinstance(analysis["tables"], list):
                analysis["tables"] = []
            if "visual_elements" not in analysis or not isinstance(analysis["visual_elements"], list):
                analysis["visual_elements"] = []
            if "document_structure" not in analysis or not isinstance(analysis["document_structure"], dict):
                analysis["document_structure"] = {}

            return {
                "page_number": page_num,
                "analysis": analysis,
                "success": True,
                "error": None
            }

        except json.JSONDecodeError as e:
            # Erreur de parsing JSON - logger le contenu reçu
            print(f"Erreur JSON parsing page {page_num}: {e}")
            print(f"Contenu reçu: {content[:500] if 'content' in locals() else 'N/A'}")
            return {
                "page_number": page_num,
                "analysis": {
                    "text_content": "",
                    "tables": [],
                    "visual_elements": [],
                    "document_structure": {}
                },
                "success": False,
                "error": f"JSON parsing error: {str(e)}"
            }

        except Exception as e:
            # Autre erreur - fallback
            print(f"Erreur analyse Pixtral page {page_num}: {e}")
            return {
                "page_number": page_num,
                "analysis": {
                    "text_content": "",
                    "tables": [],
                    "visual_elements": [],
                    "document_structure": {}
                },
                "success": False,
                "error": str(e)
            }

    def create_enriched_chunks(
        self,
        page_analyses: List[Dict[str, Any]],
        pdf_path: str,
        chunk_strategy: str = "hybrid"
    ) -> List[Document]:
        """
        Crée des chunks enrichis à partir des analyses Pixtral.

        Args:
            page_analyses: Résultats d'analyse de toutes les pages
            pdf_path: Chemin du PDF source
            chunk_strategy: "hybrid", "page_based", ou "semantic"

        Returns:
            Liste de Documents LangChain avec métadonnées enrichies
        """
        documents = []

        for page_data in page_analyses:
            if not page_data["success"]:
                continue

            page_num = page_data["page_number"]
            analysis = page_data["analysis"]

            # Vérifier que analysis est un dict
            if not isinstance(analysis, dict):
                continue

            # 1. Chunk principal: texte de la page
            main_text = analysis.get("text_content", "")

            # Convertir en string si ce n'est pas déjà le cas
            if isinstance(main_text, dict):
                # Si c'est un dict, le sérialiser en JSON
                main_text = json.dumps(main_text, ensure_ascii=False, indent=2)
            elif isinstance(main_text, list):
                # Si c'est une liste, joindre les éléments
                main_text = "\n".join(str(item) for item in main_text)
            elif not isinstance(main_text, str):
                # Convertir en string
                main_text = str(main_text)

            if main_text.strip():
                doc = Document(
                    page_content=main_text,
                    metadata={
                        "source": pdf_path,
                        "type": "pdf",
                        "page": page_num,
                        "processing": "pixtral_vision",
                        "has_tables": len(analysis.get("tables", [])) > 0,
                        "has_visuals": len(analysis.get("visual_elements", [])) > 0,
                        "layout": analysis.get("document_structure", {}).get("layout_type", "unknown"),
                        "chunk_type": "main_text"
                    }
                )
                documents.append(doc)

            # 2. Chunks pour tableaux (avec descriptions enrichies)
            for idx, table in enumerate(analysis.get("tables", [])):
                # Vérifier que table est bien un dict
                if not isinstance(table, dict):
                    continue

                headers = table.get('headers', [])
                if isinstance(headers, list):
                    headers_str = ', '.join(str(h) for h in headers)
                else:
                    headers_str = str(headers)

                # Convertir description et summary en strings
                description = str(table.get('description', ''))
                summary = str(table.get('data_summary', ''))

                table_text = f"""TABLE {idx + 1} (Page {page_num}):
Description: {description}
Headers: {headers_str}
Summary: {summary}"""

                doc = Document(
                    page_content=table_text,
                    metadata={
                        "source": pdf_path,
                        "type": "pdf",
                        "page": page_num,
                        "processing": "pixtral_vision",
                        "chunk_type": "table",
                        "table_index": idx,
                        "table_headers": headers if isinstance(headers, list) else []
                    }
                )
                documents.append(doc)

            # 3. Chunks pour éléments visuels (descriptions générées par Pixtral)
            for idx, visual in enumerate(analysis.get("visual_elements", [])):
                # Vérifier que visual est bien un dict
                if not isinstance(visual, dict):
                    continue

                # Convertir tous les champs en strings
                visual_type = str(visual.get('type', 'unknown'))
                visual_position = str(visual.get('position', 'unknown'))
                visual_description = str(visual.get('description', ''))

                visual_text = f"""VISUAL ELEMENT {idx + 1} (Page {page_num}):
Type: {visual_type}
Position: {visual_position}
Description: {visual_description}"""

                doc = Document(
                    page_content=visual_text,
                    metadata={
                        "source": pdf_path,
                        "type": "pdf",
                        "page": page_num,
                        "processing": "pixtral_vision",
                        "chunk_type": "visual",
                        "visual_type": visual_type,
                        "visual_index": idx
                    }
                )
                documents.append(doc)

        return documents

    def process_pdf_complete(
        self,
        pdf_path: str,
        dpi: int = 200,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Document]:
        """
        Pipeline complet de traitement d'un PDF avec Pixtral.

        Args:
            pdf_path: Chemin du fichier PDF
            dpi: Résolution pour conversion (200 = performance, 300 = qualité)
            progress_callback: Fonction appelée pour chaque page (optionnel)

        Returns:
            Liste de Documents enrichis prêts pour Qdrant
        """
        # 1. Conversion PDF -> Images
        images = self.convert_pdf_to_images(pdf_path, dpi=dpi)

        # 2. Analyse de chaque page avec Pixtral
        page_analyses = []
        for idx, image in enumerate(images):
            if progress_callback:
                progress_callback(idx + 1, len(images))

            analysis = self.analyze_page_with_pixtral(image, idx)
            page_analyses.append(analysis)

            # Cache optionnel des images
            if self.cache_images:
                cache_path = self.cache_dir / f"{Path(pdf_path).stem}_page_{idx}.png"
                image.save(cache_path)

        # 3. Création de chunks enrichis
        documents = self.create_enriched_chunks(page_analyses, pdf_path)

        # Si aucun document n'a été créé (toutes les pages ont échoué), lever une exception
        if not documents:
            failed_pages = [p for p in page_analyses if not p["success"]]
            if failed_pages:
                errors = [p["error"] for p in failed_pages[:3]]  # Montrer les 3 premières erreurs
                raise ValueError(f"Échec analyse Pixtral pour toutes les pages. Erreurs: {errors}")

        return documents
