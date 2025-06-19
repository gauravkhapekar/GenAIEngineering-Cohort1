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
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create shoes table from Hugging Face dataset
    # create_shoes_table_from_hf(
    #     database=".lancedb_shoes",
    #     table_name="myntra_shoes_enhanced",
    #     sample_size=500,
    #     save_images=True
    # )

    parser = argparse.ArgumentParser(description="Run RAG pipeline for shoes with text and image search")
    parser.add_argument("--query", type=str, help="Search query (text) or image file path")
    parser.add_argument("--basic-prompts", action="store_true", help="Use basic prompts instead of advanced")
    parser.add_argument("--search-type", choices=["auto", "text", "image"], default="auto", 
                       help="Force search type (default: auto-detect)")
    args = parser.parse_args()
    
    if not args.query:
        print("❌ Please provide a query using --query")
        print("\nExample usage:")
        print("  # Text search")
        print("  python script.py --query 'recommend running shoes for men'")
        print("  # Image search (auto-detected)")
        print("  python script.py --query 'path/to/shoe_image.jpg'")
        print("  # Force image search")
        print("  python script.py --query 'shoe_image.jpg' --search-type image")
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