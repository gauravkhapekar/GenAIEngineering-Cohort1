import os
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from random import sample
import re
from typing import Any, Optional, List, Dict
import torch
from datasets import load_dataset
from enum import Enum

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
from transformers import AutoTokenizer, AutoModelForCausalLM

import gradio as gr


# ============================================================================
# SECTION 1: RETRIEVAL - Vector Search and Data Management
# ============================================================================

def register_model(model_name: str) -> Any:
    """Register a model with the given name using LanceDB's EmbeddingFunctionRegistry."""
    registry = EmbeddingFunctionRegistry.get_instance()
    model = registry.get(model_name).create()
    return model


# Register the OpenAI CLIP model for vector embeddings
clip = register_model("open-clip")


class MyntraShoesEnhanced(LanceModel):
    """Enhanced Myntra Shoes Schema with product metadata for vector storage."""
    
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()
    
    # Core product information extracted from text
    product_id: Optional[str] = None
    description: Optional[str] = None
    product_type: Optional[str] = None
    gender: Optional[str] = None
    color: Optional[str] = None
    
    # Shoe-specific attributes
    toe_shape: Optional[str] = None
    pattern: Optional[str] = None
    fastening: Optional[str] = None
    shoe_width: Optional[str] = None
    ankle_height: Optional[str] = None
    insole: Optional[str] = None
    sole_material: Optional[str] = None

    @property
    def image(self):
        if isinstance(self.image_uri, str) and os.path.exists(self.image_uri):
            return Image.open(self.image_uri)
        elif hasattr(self.image_uri, 'save'):  # PIL Image object
            return self.image_uri
        else:
            # Return a placeholder or handle the case appropriately
            return None


def parse_shoe_attributes(text: str) -> dict:
    """Parse shoe attributes from the text description for structured storage."""
    attributes = {}
    
    # Extract product type (Men/Women + product type)
    if text.startswith('Men '):
        attributes['gender'] = 'Men'
        attributes['product_type'] = text.split('Men ')[1].split('.')[0].strip()
    elif text.startswith('Women '):
        attributes['gender'] = 'Women'
        attributes['product_type'] = text.split('Women ')[1].split('.')[0].strip()
    else:
        attributes['gender'] = None
        attributes['product_type'] = text.split('.')[0].strip()
    
    # Extract structured attributes using regex
    patterns = {
        'toe_shape': r'Toe Shape: ([^,]+)',
        'pattern': r'Pattern: ([^,]+)',
        'fastening': r'Fastening: ([^,]+)',
        'shoe_width': r'Shoe Width: ([^,]+)',
        'ankle_height': r'Ankle Height: ([^,]+)',
        'insole': r'Insole: ([^,]+)',
        'sole_material': r'Sole Material: ([^,\.]+)'
    }
    
    for attr, pattern in patterns.items():
        match = re.search(pattern, text)
        attributes[attr] = match.group(1).strip() if match else None
    
    # Extract color information (basic color detection)
    color_keywords = ['white', 'black', 'brown', 'blue', 'red', 'green', 'grey', 'gray', 
                     'navy', 'tan', 'beige', 'pink', 'purple', 'yellow', 'orange']
    
    text_lower = text.lower()
    detected_colors = [color for color in color_keywords if color in text_lower]
    attributes['color'] = ', '.join(detected_colors) if detected_colors else None
    
    return attributes


def create_shoes_table_from_hf(
    database: str,
    table_name: str,
    dataset_name: str = "Harshgarg12/myntra_shoes_dataset",
    schema: Any = MyntraShoesEnhanced,
    mode: str = "overwrite",
    sample_size: int = 500,
    save_images: bool = True,
    images_dir: str = "hf_shoe_images"
) -> None:
    """Create vector database table with shoe data from Hugging Face dataset."""
    
    db = lancedb.connect(database)
    
    if table_name in db and mode != "overwrite":
        print(f"Table {table_name} already exists")
        return
    
    # Load dataset from Hugging Face
    print("Loading dataset from Hugging Face...")
    ds = load_dataset(dataset_name)
    train_data = ds['train']
    
    # Sample data if needed
    if len(train_data) > sample_size:
        indices = sample(range(len(train_data)), sample_size)
        train_data = train_data.select(indices)
    
    print(f"Processing {len(train_data)} samples...")
    
    # Create images directory if saving images
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    
    # Prepare data for table creation
    table_data = []
    for i, item in enumerate(train_data):
        image = item['image']
        text = item['text']
        
        # Parse attributes from text
        attributes = parse_shoe_attributes(text)
        
        # Handle image
        if save_images:
            image_path = os.path.join(images_dir, f"shoe_{i:04d}.jpg")
            image.save(image_path, "JPEG")
            image_uri = image_path
        else:
            # Store PIL image directly (may cause issues with serialization)
            image_uri = image
        
        table_data.append({
            'image_uri': image_uri,
            'product_id': f"hf_shoe_{i:04d}",
            'description': text,
            'product_type': attributes.get('product_type'),
            'gender': attributes.get('gender'),
            'color': attributes.get('color'),
            'toe_shape': attributes.get('toe_shape'),
            'pattern': attributes.get('pattern'),
            'fastening': attributes.get('fastening'),
            'shoe_width': attributes.get('shoe_width'),
            'ankle_height': attributes.get('ankle_height'),
            'insole': attributes.get('insole'),
            'sole_material': attributes.get('sole_material'),
        })
    
    if table_data:
        if table_name in db:
            db.drop_table(table_name)
        
        table = db.create_table(table_name, schema=schema, mode="create")
        table.add(pd.DataFrame(table_data))
        print(f"Added {len(table_data)} shoes to table")
    else:
        print("No data to add")


def run_shoes_search(
    database: str,
    table_name: str,
    schema: Any,
    search_query: Any,
    limit: int = 6,
    output_folder: str = "shoe_search_output",
    search_type: str = "auto"  # "auto", "text", "image"
) -> tuple[list, str]:
    """RETRIEVAL: Run vector search on shoes and return detailed results."""
    
    # Clean output folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)
    
    db = lancedb.connect(database)
    table = db.open_table(table_name)
    
    # Determine search type and process query
    actual_search_type = search_type
    processed_query = search_query
    
    if search_type == "auto":
        # Auto-detect search type
        if isinstance(search_query, str):
            if search_query.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                # Image file path
                try:
                    processed_query = Image.open(search_query)
                    actual_search_type = "image"
                    print(f"🖼️  Detected image search: {search_query}")
                except Exception as e:
                    print(f"❌ Error loading image: {e}")
                    return [], "error"
            else:
                # Text query
                actual_search_type = "text"
                print(f"📝 Detected text search: {search_query}")
        elif hasattr(search_query, 'save'):  # PIL Image object
            actual_search_type = "image"
            processed_query = search_query
            print("🖼️  Detected image search: PIL Image object")
        else:
            actual_search_type = "text"
            print(f"📝 Detected text search: {search_query}")
    
    elif search_type == "image":
        if isinstance(search_query, str):
            try:
                processed_query = Image.open(search_query)
                print(f"🖼️  Image search: {search_query}")
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                return [], "error"
        elif hasattr(search_query, 'save'):
            processed_query = search_query
            print("🖼️  Image search: PIL Image object")
        else:
            print("❌ Invalid image input for image search")
            return [], "error"
    
    else:  # text search
        actual_search_type = "text"
        print(f"📝 Text search: {search_query}")
    
    # Perform vector search
    try:
        results = table.search(processed_query).limit(limit).to_pydantic(schema)
    except Exception as e:
        print(f"❌ Search error: {e}")
        return [], "error"
    
    # Save images and collect metadata
    search_results = []
    for i, result in enumerate(results):
        image_path = os.path.join(output_folder, f"result_{i}.jpg")
        
        # Handle different image storage methods
        if result.image:
            result.image.save(image_path, "JPEG")
        else:
            print(f"Warning: No image available for result {i}")
            continue
        
        search_results.append({
            'rank': i + 1,
            'product_id': result.product_id,
            'description': result.description[:100] + "..." if result.description and len(result.description) > 100 else result.description,
            'product_type': result.product_type,
            'gender': result.gender,
            'color': result.color,
            'toe_shape': result.toe_shape,
            'pattern': result.pattern,
            'fastening': result.fastening,
            'image_path': image_path
        })
    
    return search_results, actual_search_type





# ============================================================================
# SECTION 2: AUGMENTATION - Context Enhancement and Prompt Engineering
# ============================================================================

class QueryType(Enum):
    """Query types for different shoe-related interactions."""
    RECOMMENDATION = "recommendation"
    SEARCH = "search"


class SimpleShoePrompts:
    """AUGMENTATION: Simplified prompt system for shoe RAG with context enhancement."""
    
    def __init__(self):
        self.system_prompts = {
            "recommendation": """You are a helpful assistant. Choose from the given shoe options and give a short, simple recommendation. Do not make up any information.""",
            
            "search": """You are a knowledgeable shoe assistant. Help customers understand the available shoe options 
that match their search criteria, providing detailed information about features and benefits."""
        }
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query into recommendation or search type."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['recommend', 'suggest', 'best', 'need', 'looking for']):
            return QueryType.RECOMMENDATION
        else:
            return QueryType.SEARCH
    
    def format_shoes_context(self, shoes: List[Dict[str, Any]]) -> str:
        """AUGMENTATION: Format retrieved shoes into readable context for LLM."""
        formatted_shoes = []
        for i, shoe in enumerate(shoes, 1):
            # Keep it simple - just basic info
            product_type = shoe.get('product_type', 'Shoe')
            gender = shoe.get('gender', '')
            
            if gender:
                shoe_name = f"{product_type} for {gender}"
            else:
                shoe_name = product_type
            
            # Add basic color info if available
            color = shoe.get('color', '')
            if color and color not in ['None', None, '']:
                shoe_name += f" ({color})"
            
            formatted_shoes.append(f"{i}. {shoe_name}")
        
        return "\n".join(formatted_shoes)
    
    def generate_prompt(self, query: str, shoes: List[Dict[str, Any]], search_type: str = "text") -> str:
        """AUGMENTATION: Generate complete prompt based on query type and retrieved context."""
        # If it's an image search, always treat as search query type
        if search_type == "image":
            query_type = QueryType.SEARCH
        else:
            query_type = self.classify_query(query)
            
        system_prompt = self.system_prompts[query_type.value]
        context = self.format_shoes_context(shoes)
        
        if query_type == QueryType.RECOMMENDATION:
            # Add a summary to guide recommendations
            intent_summary = f"Based on the query, the user is likely looking for {query.lower()}."

            user_prompt = f"""{intent_summary}

    Available Options:
    {context}

    Your task:
    - Recommend the best option(s) that align most closely with the query.
    - Reference specific attributes (e.g., gender, product type, color, or other features) in your reasoning.
    - Avoid adding details not provided in the context.

    Provide your recommendation in 2-3 sentences."""
        
        else:  # SEARCH
            user_prompt = f"""Here are shoes matching: "{query}"

    Search Results:
    {context}

    Explain how well these shoes meet the search criteria and highlight their relevant features."""
        
        return f"{system_prompt}\n\n{user_prompt}"




# ============================================================================
# SECTION 3: GENERATION - LLM Setup and Response Generation
# ============================================================================

def setup_qwen_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> tuple:
    """GENERATION: Setup Qwen2.5-0.5B model for text generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    return tokenizer, model


def generate_shoes_rag_response(
    tokenizer,
    model,
    query: str,
    retrieved_shoes: List[Dict[str, Any]],
    max_tokens: int = 600,
    use_advanced_prompts: bool = True
) -> str:
    """GENERATION: Generate RAG response using retrieved shoes context with prompt engineering."""
    
    if use_advanced_prompts:
        # Use the simplified prompt system
        prompt_manager = SimpleShoePrompts()
        complete_prompt = prompt_manager.generate_prompt(query, retrieved_shoes)
        query_type = prompt_manager.classify_query(query)
        print(f"Using {query_type.value} prompt for query")
        
    else:
        # Use the basic prompt system (fallback)
        prompt_manager = SimpleShoePrompts()
        context = prompt_manager.format_shoes_context(retrieved_shoes)
        complete_prompt = f"""Based on the following shoe products, answer the user's question:

Shoes:
{context}

Question: {query}

Answer:"""
    
    inputs = tokenizer(complete_prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Ensure everything runs on CPU
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,  # Very low temperature to reduce hallucination
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,  # Minimal repetition penalty
            no_repeat_ngram_size=2,
            early_stopping=False
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# COMPLETE RAG PIPELINE - Integrating All Three Sections
# ============================================================================

def run_complete_shoes_rag_pipeline(
    database: str,
    table_name: str,
    schema: Any,
    search_query: Any,  # Can be text string or image path/PIL Image
    limit: int = 3,
    use_llm: bool = True,
    use_advanced_prompts: bool = True,
    search_type: str = "auto"
) -> Dict[str, Any]:
    """Run complete RAG pipeline integrating Retrieval, Augmentation, and Generation."""
    
    # SECTION 1: RETRIEVAL - Get relevant shoes from vector database
    print("🔍 RETRIEVAL: Searching for relevant shoes...")
    results, actual_search_type = run_shoes_search(database, table_name, schema, search_query, limit, search_type=search_type)
    
    if not results:
        return {"query": search_query, "results": [], "response": "No results found", "search_type": actual_search_type}
    
    if not use_llm:
        return {"query": search_query, "results": results, "response": None, "search_type": actual_search_type}
    
    # SECTION 2: AUGMENTATION - Process and enhance context with prompt engineering
    try:
        print("📝 AUGMENTATION: Enhancing context with prompt engineering...")
        
        # Set up prompt manager and analyze query
        prompt_manager = SimpleShoePrompts()
        
        # For image search, use appropriate query text
        if actual_search_type == "image":
            query_text = "similar shoes based on the provided image"
            print(f"   └─ Image search - using search query type")
        else:
            query_text = str(search_query)
            query_type = prompt_manager.classify_query(query_text)
            print(f"   └─ Text query classified as: {query_type.value}")
        
        # Format context and generate enhanced prompt
        enhanced_prompt = prompt_manager.generate_prompt(query_text, results, actual_search_type)
        print(f"   └─ Context formatted with {len(results)} retrieved shoes")
        
        # SECTION 3: GENERATION - Setup LLM and generate response
        print("🤖 GENERATION: Setting up LLM and generating response...")
        tokenizer, model = setup_qwen_model()
        print("   └─ Model loaded successfully")
        
        # Generate final response using augmented context
        response = generate_shoes_rag_response(
            tokenizer, model, query_text, results, 
            max_tokens=200,
            use_advanced_prompts=use_advanced_prompts
        )
        
        # Add prompt analysis
        if actual_search_type == "image":
            final_query_type = QueryType.SEARCH.value
        else:
            final_query_type = query_type.value
            
        prompt_analysis = {
            'query_type': final_query_type,
            'num_results': len(results),
            'search_type': actual_search_type
        }
        
        return {
            "query": search_query, 
            "results": results, 
            "response": response,
            "prompt_analysis": prompt_analysis,
            "search_type": actual_search_type
        }
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return {"query": search_query, "results": results, "response": "LLM unavailable - showing search results only", "search_type": actual_search_type}





# ============================================================================
# GRADIO INTERFACE - Web App for RAG Pipeline
# ============================================================================

def gradio_rag_pipeline(query, image, search_type, use_advanced_prompts):
    """Gradio interface function for RAG pipeline."""
    try:
        # Determine the actual query based on inputs
        if search_type == "image" and image is not None:
            actual_query = image
        elif search_type == "text" and query.strip():
            actual_query = query
        elif search_type == "auto":
            if image is not None:
                actual_query = image
            elif query.strip():
                actual_query = query
            else:
                return "❌ Please provide either a text query or upload an image.", "", None, None, None
        else:
            return "❌ Please provide appropriate input for the selected search type.", "", None, None, None
        
        # Run the RAG pipeline
        rag_result = run_complete_shoes_rag_pipeline(
            database=".lancedb_shoes",
            table_name="myntra_shoes_enhanced",
            schema=MyntraShoesEnhanced,
            search_query=actual_query,
            limit=3,
            use_llm=True,
            use_advanced_prompts=use_advanced_prompts,
            search_type=search_type
        )
        
        # Format the response
        response = rag_result.get('response', 'No response generated')
        search_type_used = rag_result.get('search_type', 'unknown')
        
        # Format results for display
        results_text = f"🔍 Search Type: {search_type_used}\n\n"
        if rag_result.get('prompt_analysis'):
            results_text += f"📝 Query Type: {rag_result['prompt_analysis']['query_type']}\n"
            results_text += f"📊 Results Found: {rag_result['prompt_analysis']['num_results']}\n\n"
        
        # Prepare image gallery data
        image_gallery = []
        results_details = []
        
        for i, result in enumerate(rag_result['results'], 1):
            product_type = result.get('product_type', 'Shoe')
            gender = result.get('gender', 'Unisex')
            color = result.get('color', 'Various colors')
            pattern = result.get('pattern', 'Standard')
            description = result.get('description', 'No description available')
            image_path = result.get('image_path')
            
            # Add to gallery if image exists
            if image_path and os.path.exists(image_path):
                # Create detailed caption for the image
                caption = f"#{i} - {product_type} for {gender}"
                if color and color not in ['None', None, '']:
                    caption += f" | Color: {color}"
                if pattern and pattern not in ['None', None, '']:
                    caption += f" | Pattern: {pattern}"
                
                image_gallery.append((image_path, caption))
            
            # Format detailed description
            detail_text = f"**{i}. {product_type} for {gender}**\n"
            detail_text += f"   • Color: {color}\n"
            detail_text += f"   • Pattern: {pattern}\n"
            if description:
                # Truncate description for readability
                short_desc = description[:150] + "..." if len(description) > 150 else description
                detail_text += f"   • Description: {short_desc}\n"
            detail_text += "\n"
            results_details.append(detail_text)
        
        # Combine all details
        formatted_results = "".join(results_details)
        
        return response, formatted_results, image_gallery
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", []

def create_gradio_app():
    """Create and launch the Gradio application."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 20px !important;
        background: #f5f5f5;
    }
    .main {
        max-width: 100% !important;
        width: 100% !important;
    }
    /* Header styling */
    .header-section {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-section h1 {
        font-size: 2.5em;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-section p {
        font-size: 1.2em;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    .output-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
    }
    .gallery-container {
        margin-top: 15px;
        width: 100%;
    }
    .search-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        height: fit-content;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    .results-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        height: fit-content;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .gallery-section {
        background: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
        width: 100%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    /* Improve text readability */
    .gradio-textbox textarea {
        background: #fafafa !important;
        border: 1px solid #ddd !important;
        color: #333 !important;
    }
    .gradio-textbox textarea:focus {
        background: #ffffff !important;
        border-color: #667eea !important;
    }
    /* Make gallery images larger */
    .gallery img {
        max-height: 300px !important;
        object-fit: contain !important;
    }
    /* Improve button styling */
    .primary-button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    /* Section headers */
    .section-header {
        color: #667eea;
        font-weight: bold;
        border-bottom: 2px solid #667eea;
        padding-bottom: 5px;
        margin-bottom: 15px;
    }
    """
    
    with gr.Blocks(css=css, title="👟 Shoe RAG Pipeline") as app:
        # Header Section
        with gr.Row():
            with gr.Column(elem_classes=["header-section"]):
                gr.HTML("""
                <div style="text-align: center;">
                    <h1>👟 Multimodal Shoe RAG Pipeline</h1>
                    <p>This demo showcases a complete <strong>Retrieval-Augmented Generation (RAG)</strong> pipeline for shoe recommendations and search.</p>
                    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px; flex-wrap: wrap;">
                        <div>🔍 <strong>Text Search</strong><br/>Natural language queries</div>
                        <div>🖼️ <strong>Image Search</strong><br/>Visual similarity matching</div>
                        <div>🤖 <strong>AI Recommendations</strong><br/>LLM-powered suggestions</div>
                        <div>📊 <strong>Structured Results</strong><br/>Detailed product information</div>
                    </div>
                </div>
                """)
        
        with gr.Row(equal_height=False):
            # Left Column - Search Input
            with gr.Column(scale=1, elem_classes=["search-section"]):
                gr.HTML('<h3 class="section-header">🔍 Search Input</h3>')
                
                query = gr.Textbox(
                    label="Text Query",
                    placeholder="e.g., 'Recommend running shoes for men' or 'Show me casual sneakers'",
                    lines=4,
                    max_lines=6
                )
                
                image = gr.Image(
                    label="Upload Shoe Image (for image search)",
                    type="pil",
                    height=220
                )
                
                with gr.Row():
                    search_type = gr.Radio(
                        choices=["auto", "text", "image"],
                        value="auto",
                        label="Search Type",
                        info="Auto-detect or force specific search type"
                    )
                
                use_advanced_prompts = gr.Checkbox(
                    value=True,
                    label="Use Advanced Prompts",
                    info="Enable enhanced prompt engineering for better responses"
                )
                
                search_btn = gr.Button(
                    "🔍 Search", 
                    variant="primary", 
                    size="lg",
                    elem_classes=["primary-button"]
                )
            
            # Right Column - Results
            with gr.Column(scale=2, elem_classes=["results-section"]):
                gr.HTML('<h3 class="section-header">🤖 AI Response</h3>')
                response_output = gr.Textbox(
                    label="RAG Response",
                    lines=6,
                    max_lines=10,
                    elem_classes=["output-text"],
                    show_copy_button=True
                )
                
                gr.HTML('<h3 class="section-header">📊 Search Results Details</h3>')
                results_output = gr.Textbox(
                    label="Product Information",
                    lines=8,
                    max_lines=12,
                    elem_classes=["output-text"],
                    show_copy_button=True
                )
        
        # Full width section for image gallery
        with gr.Row():
            with gr.Column(elem_classes=["gallery-section"]):
                gr.HTML('<h3 class="section-header">🖼️ Retrieved Shoe Images</h3>')
                image_gallery = gr.Gallery(
                    label="Search Results Gallery",
                    show_label=False,
                    elem_id="gallery",
                    columns=3,
                    rows=1,
                    object_fit="contain",
                    height=350,
                    elem_classes=["gallery-container"],
                    preview=True
                )
        
        # Event handlers
        search_btn.click(
            fn=gradio_rag_pipeline,
            inputs=[query, image, search_type, use_advanced_prompts],
            outputs=[response_output, results_output, image_gallery]
        )
        
    return app

# ============================================================================
# MAIN EXECUTION WITH GRADIO OPTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline for shoes with text and image search")
    parser.add_argument("--query", type=str, help="Search query (text) or image file path")
    parser.add_argument("--basic-prompts", action="store_true", help="Use basic prompts instead of advanced")
    parser.add_argument("--search-type", choices=["auto", "text", "image"], default="auto", 
                       help="Force search type (default: auto-detect)")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio web interface")
    parser.add_argument("--setup-db", action="store_true", help="Setup database from HuggingFace dataset")
    args = parser.parse_args()
    
    # Setup database if requested
    if args.setup_db:
        print("🔄 Setting up database from HuggingFace dataset...")
        create_shoes_table_from_hf(
            database=".lancedb_shoes",
            table_name="myntra_shoes_enhanced",
            sample_size=500,
            save_images=True
        )
        print("✅ Database setup complete!")
        exit(0)
    
    # Launch Gradio interface if requested
    if args.gradio:
        print("🚀 Launching Gradio interface...")
        app = create_gradio_app()
        app.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7862,
            show_error=True
        )
        exit(0)
    
    # Command line interface
    if not args.query:
        print("❌ Please provide a query using --query, or use --gradio for web interface")
        print("\nExample usage:")
        print("  # Command line search")
        print("  python script.py --query 'recommend running shoes for men'")
        print("  # Launch web interface")
        print("  python script.py --gradio")
        print("  # Setup database first")
        print("  python script.py --setup-db")
        exit(1)
    
    # Single query processing
    print("🚀 Starting Complete RAG Pipeline...")
    print("=" * 60)
    
    rag_result = run_complete_shoes_rag_pipeline(
        database=".lancedb_shoes",
        table_name="myntra_shoes_enhanced",
        schema=MyntraShoesEnhanced,
        search_query=args.query,
        limit=3,
        use_llm=True,
        use_advanced_prompts=not args.basic_prompts,
        search_type=args.search_type
    )
    
    # Display results
    print("=" * 60)
    print("📊 RAG PIPELINE RESULTS")
    print("=" * 60)
    print(f"Query: {rag_result['query']}")
    print(f"Search Type: {rag_result['search_type']}")
    if rag_result.get('prompt_analysis'):
        print(f"Query Type: {rag_result['prompt_analysis']['query_type']}")
        print(f"Results Found: {rag_result['prompt_analysis']['num_results']}")
    print(f"\n💬 RAG Response: {rag_result['response']}")
    print(f"\n👟 Retrieved Shoes:")
    for result in rag_result['results']:
        print(f"- {result['product_type']} ({result['gender']}) - {result['color']} - {result['pattern']}")
        if rag_result['search_type'] == 'image':
            print(f"  📁 Image saved: {result['image_path']}")
    
    if rag_result['search_type'] == 'image':
        print(f"\n🖼️  Search results images saved in: shoe_search_output/")

    ### Sample search queries
    # Recommend me office shoes for men
    # Recommend me casual sneakers for men
    # Show me casual high top white sneakers for men 
    # Show me comfortable running shoes for men
    # Show me slippers for men 