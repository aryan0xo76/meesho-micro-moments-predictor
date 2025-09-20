"""
FastAPI Backend for Meesho Micro-Moment Prediction Engine
CUDA 11.8 compatible, GA4/GSC data generation, CatBoost GPU training
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import our custom modules
from data_generator import DataGenerator
from models.covisitation import CovisitationModel
from models.send_time_optimization import SendTimeOptimizer
from models.reranker import CatBoostReranker
from models.headline_generator import HeadlineGenerator

# Initialize FastAPI app
app = FastAPI(
    title="Meesho Micro-Moment Prediction Engine",
    description="AI-Powered WhatsApp Commerce Optimization",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base paths for static
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"

# Global models storage
models = {
    "covisitation": None,
    "sto": None,
    "reranker": None,
    "headlines": None
}

# Initialize models with default instances
def initialize_models():
    """Initialize models with default instances"""
    global models
    if models["headlines"] is None:
        models["headlines"] = HeadlineGenerator()
    if models["covisitation"] is None:
        models["covisitation"] = CovisitationModel()
    if models["sto"] is None:
        models["sto"] = SendTimeOptimizer()
    if models["reranker"] is None:
        models["reranker"] = CatBoostReranker()

# Initialize models on startup
initialize_models()

# Training status
training_status = {"status": "idle", "progress": 0, "message": ""}

# Personas configuration - UPDATED with correct categories
PERSONAS = [
    {
        "id": "tier2_fashion",
        "name": "Tier-2 Fashion Family Shopper",
        "description": "Women's ethnic wear focus, evening engagement peak, Diwali/wedding season boost",
        "preferred_categories": ["fashion", "ethnic", "jewelry", "accessories"],
        "peak_hours": [20, 21, 22],
        "region_focus": "tier2_cities",
        "seasonal_boost": ["Diwali", "Wedding Season", "Navratri"]
    },
    {
        "id": "student_examprep",
        "name": "Campus Student Exam-Prep", 
        "description": "Stationery and budget electronics focus, late-night engagement, exam window boost",
        "preferred_categories": ["books", "stationery", "electronics"],
        "peak_hours": [22, 23, 0, 1],
        "region_focus": "college_towns",
        "seasonal_boost": ["Board Exams", "Competitive Exams"]
    },
    {
        "id": "budget_gadget",
        "name": "Budget Gadget Seeker",
        "description": "Low-cost electronics, weekend timing, festival deal sensitivity", 
        "preferred_categories": ["electronics", "gadgets", "mobile", "accessories"],
        "peak_hours": [18, 19, 20, 21],
        "region_focus": "urban_suburban", 
        "seasonal_boost": ["Diwali Sale", "Republic Day Sale"]
    },
    {
        "id": "home_decor_festive",
        "name": "Home Decor Festive Upgrader",
        "description": "Decor and lighting before Diwali, afternoon engagement pattern",
        "preferred_categories": ["accessories", "jewelry", "puja_items"],
        "peak_hours": [14, 15, 16, 17],
        "region_focus": "metro_suburban",
        "seasonal_boost": ["Pre-Diwali", "Gudi Padwa", "House Warming"]
    },
    {
        "id": "regional_festive", 
        "name": "Regional Festive Wear (Bengal Focus)",
        "description": "Sarees and puja items during Durga Puja, morning browsing pattern",
        "preferred_categories": ["fashion", "ethnic", "saree"],
        "peak_hours": [7, 8, 9, 10],
        "region_focus": "west_bengal",
        "seasonal_boost": ["Durga Puja", "Kali Puja", "Poila Boishakh"]
    }
]


# Request/Response models
class DataGenerationRequest(BaseModel):
    n_users: int = 8000
    n_products: int = 3000
    days: int = 120
    seed: int = 42

class TrainingRequest(BaseModel):
    models: List[str] = ["covisitation", "sto", "reranker", "headlines"]

class RecommendationRequest(BaseModel):
    persona_id: str
    date_range: Optional[str] = None

class RecommendationResponse(BaseModel):
    headline: str
    optimal_hours: List[int]
    products: List[Dict[str, Any]]
    whatsapp_message: str
    persona_info: Dict[str, Any]

# Utility functions
def ensure_data_dir():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_persona_by_id(persona_id: str) -> Optional[Dict]:
    return next((p for p in PERSONAS if p["id"] == persona_id), None)

def filter_products_by_persona(df_products: pd.DataFrame, persona: Dict) -> pd.DataFrame:
    """Filter products based on persona preferences with flexible matching"""
    if df_products.empty:
        raise HTTPException(status_code=500, detail="No product data available. Generate data first.")

    print(f"Loaded {len(df_products)} products, {len(df_events)} events") 
    
    persona_categories = [cat.lower() for cat in persona["preferred_categories"]]
    
    # Try exact category matches first
    exact_matches = df_products[
        df_products["category"].str.lower().isin(persona_categories)
    ]
    
    if not exact_matches.empty:
        return exact_matches
    
    # Try partial matches in category or title
    partial_matches = df_products[
        df_products["category"].str.lower().str.contains("|".join(persona_categories), na=False) |
        df_products["title"].str.lower().str.contains("|".join(persona_categories), na=False)
    ]
    
    return partial_matches if not partial_matches.empty else df_products.head(10)

#test
@app.post("/api/debug-generate")
async def debug_generate():
    try:
        from data_generator import DataGenerator
        generator = DataGenerator()
        
        # Test each step individually
        results = {}
        
        # Step 1: Test data directory creation
        import os
        os.makedirs("data", exist_ok=True)
        results["data_dir"] = "OK"
        
        # Step 2: Test small data generation
        files = await generator.generate_all_data(n_users=10, n_products=5, days=7, seed=42)
        results["files_created"] = files
        results["status"] = "SUCCESS"
        
        return results
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }

# ---------- SPA entry (serve index.html at "/") ----------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

# ---------- API Endpoints ----------
@app.get("/api/status")
async def get_status():
    data_files = {
        "events": os.path.exists("data/events.csv"),
        "products": os.path.exists("data/products.csv"),
        "users": os.path.exists("data/users.csv"),
        "calendar": os.path.exists("data/calendar.csv"),
        "gsc_queries": os.path.exists("data/gsc_queries.csv")
    }
    model_files = {
        "covisitation": os.path.exists("data/covisitation_model.pkl"),
        "sto": os.path.exists("data/sto_model.pkl"),
        "reranker": os.path.exists("data/reranker.cbm"),
        "headlines": os.path.exists("data/headlines_model_info.json")
    }
    return {
        "status": "healthy",
        "data_generated": all(data_files.values()),
        "models_trained": any(model_files.values()),
        "data_files": data_files,
        "model_files": model_files,
        "training_status": training_status
    }

@app.get("/api/personas")
async def get_personas():
    return {"personas": PERSONAS}

@app.post("/api/generate-data")
async def generate_data(request: DataGenerationRequest):
    try:
        ensure_data_dir()
        generator = DataGenerator()
        files_created = await generator.generate_all_data(
            n_users=request.n_users,
            n_products=request.n_products,
            days=request.days,
            seed=request.seed
        )
        return {
            "status": "success",
            "message": f"Generated {len(files_created)} data files",
            "files": files_created,
            "parameters": request.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

async def train_models_background(models_to_train: List[str]):
    global training_status, models
    try:
        training_status = {"status": "running", "progress": 0, "message": "Starting training..."}
        df_events = pd.read_csv("data/events.csv")
        df_products = pd.read_csv("data/products.csv")
        df_users = pd.read_csv("data/users.csv")
        df_calendar = pd.read_csv("data/calendar.csv")
        total_steps = len(models_to_train)
        current_step = 0

        if "covisitation" in models_to_train:
            training_status["message"] = "Training co-visitation model..."
            training_status["progress"] = int((current_step / total_steps) * 100)
            covis_model = CovisitationModel()
            await covis_model.train(df_events)
            models["covisitation"] = covis_model
            current_step += 1

        if "sto" in models_to_train:
            training_status["message"] = "Training send-time optimizer..."
            training_status["progress"] = int((current_step / total_steps) * 100)
            sto_model = SendTimeOptimizer()
            await sto_model.train(df_events)
            models["sto"] = sto_model
            current_step += 1

        if "reranker" in models_to_train:
            training_status["message"] = "Training CatBoost re-ranker..."
            training_status["progress"] = int((current_step / total_steps) * 100)
            reranker = CatBoostReranker()
            await reranker.train(df_events, df_products)
            models["reranker"] = reranker
            current_step += 1

        if "headlines" in models_to_train:
            training_status["message"] = "Initializing headline generator..."
            training_status["progress"] = int((current_step / total_steps) * 100)
            headline_gen = HeadlineGenerator()
            await headline_gen.train(df_products, df_calendar)
            models["headlines"] = headline_gen
            current_step += 1

        training_status = {
            "status": "completed",
            "progress": 100,
            "message": f"Successfully trained {len(models_to_train)} models"
        }
    except Exception as e:
        training_status = {"status": "error", "progress": 0, "message": f"Training failed: {str(e)}"}

@app.post("/api/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    required_files = ["data/events.csv", "data/products.csv", "data/users.csv", "data/calendar.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise HTTPException(status_code=400, detail=f"Missing required data files: {missing_files}. Generate data first.")
    if training_status["status"] == "running":
        return {"status": "already_running", "message": "Training is already in progress", "training_status": training_status}
    background_tasks.add_task(train_models_background, request.models)
    return {"status": "started", "message": "Model training started in background", "models_to_train": request.models}

@app.get("/api/training-status")
async def get_training_status():
    return training_status

@app.post("/api/recommend")
async def generate_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    persona = get_persona_by_id(request.persona_id)
    if not persona:
        raise HTTPException(status_code=400, detail=f"Invalid persona_id: {request.persona_id}")
    
     # ADD FIX 4 HERE - Debug logging
    print(f"Available models: {list(models.keys())}")
    print(f"Headlines model: {models['headlines']}")
    print(f"Headlines model trained: {models['headlines'].trained if models['headlines'] else 'None'}")
    print(f"Covisitation model: {models['covisitation']}")
    print(f"STO model: {models['sto']}")
    print(f"Reranker model: {models['reranker']}")

    try:
        df_events = pd.read_csv("data/events.csv") if os.path.exists("data/events.csv") else pd.DataFrame()
        df_products = pd.read_csv("data/products.csv") if os.path.exists("data/products.csv") else pd.DataFrame()
        df_calendar = pd.read_csv("data/calendar.csv") if os.path.exists("data/calendar.csv") else pd.DataFrame()

        optimal_hours = persona["peak_hours"][:3]
        if models["sto"]:
            try:
                optimal_hours = await models["sto"].get_optimal_hours(request.persona_id, df_events)
            except Exception:
                optimal_hours = persona["peak_hours"][:3]

        # FIXED PRODUCT FILTERING LOGIC
        products = []
        if not df_products.empty:
            # Map persona categories to actual data categories
            category_mappings = {
                # Budget Gadget mappings (fix the products issue)
                "electronics": ["electronics_budget", "electronics_accessories"],
                "gadgets": ["gadgets_budget"], 
                "mobile": ["mobile_accessories"],
                "accessories": ["accessories", "mobile_accessories", "electronics_accessories"],
                
                # Home Decor mappings (fix the HTTP error)
                "homedecor": ["homedecor"],
                "decor": ["homedecor"], 
                "lighting": ["lighting"],
                "furnishing": ["furnishing"],
                
                # Other mappings
                "fashion": ["fashion_ethnic", "designer_wear"],
                "ethnic": ["fashion_ethnic", "ethnic_bengali"],
                "stationery": ["stationery"],
                "books": ["books"],
                "jewelry": ["jewelry"]
            }


            
            # Build list of actual categories to search for
            search_categories = []
            for pref_cat in persona["preferred_categories"]:
                if pref_cat in category_mappings:
                    mapping = category_mappings[pref_cat]
                    if isinstance(mapping, list):
                        search_categories.extend(mapping)
                    else:
                        search_categories.append(mapping)
                else:
                    search_categories.append(pref_cat)
            
            # Filter products by mapped categories
            filtered_products = df_products[
                df_products["category"].isin(search_categories)
            ]

            #logging
            print(f"Persona categories: {persona['preferred_categories']}")
            print(f"Search categories: {search_categories}")  
            print(f"Filtered products count: {len(filtered_products)}")
            print(f"Available categories in data: {df_products['category'].unique()}")
            print(f"First few filtered products: {filtered_products.head()['category'].tolist()}")
            
            # If no exact matches, try partial matches
            if filtered_products.empty:
                mask = df_products["category"].str.contains(
                    "|".join(search_categories), case=False, na=False
                )
                filtered_products = df_products[mask]
            
            # If still empty, try title matching
            if filtered_products.empty:
                mask = df_products["title"].str.contains(
                    "|".join(persona["preferred_categories"]), case=False, na=False
                )
                filtered_products = df_products[mask]
            
            # Use ML model if available, otherwise random sample
            if models["covisitation"] and hasattr(models["covisitation"], 'trained') and models["covisitation"].trained and not filtered_products.empty:
                try:
                    products = await models["covisitation"].get_recommendations(
                        persona["preferred_categories"], filtered_products, limit=6
                    )
                except Exception:
                    # Trained model failed, fallback to sampling
                    if len(filtered_products) >= 6:
                        products = filtered_products.sample(n=6, random_state=42).to_dict("records")
                    else:
                        products = filtered_products.to_dict("records")
            elif not filtered_products.empty:
                # No trained model available, use category filtering + sampling
                if len(filtered_products) >= 6:
                    products = filtered_products.sample(n=6, random_state=42).to_dict("records") 
                else:
                    products = filtered_products.to_dict("records")


        
        # Ensure we have exactly 6 products for frontend
        if len(products) < 6 and not df_products.empty:
            remaining = 6 - len(products)
            used_ids = [p.get('productid', p.get('product_id', 0)) for p in products]
            extra_products = df_products[~df_products['productid'].isin(used_ids)].sample(n=min(remaining, len(df_products) - len(used_ids))).to_dict("records")
            products.extend(extra_products)
        
        products = products[:6]  # Ensure exactly 6

        headline = "🌟 Special Picks: Just for You! 🌟"
        if models["headlines"]:
            try:
                headline = await models["headlines"].generate_headline(persona, df_calendar)
                print(f"Generated headline: {headline}")  # Add logging
            except Exception as e:
                print(f"Headline generation failed: {e}")  # Log the actual error
                import traceback
                traceback.print_exc()

        product_list = "\\n".join([
            f"• {p.get('title', 'Product')} - ₹{int(p.get('price', 0))}"
            for p in products[:5]
        ]) if products else "• Great products coming soon!"

        whatsapp_message = f"""{headline}

{product_list}

Best time to share: {', '.join([f'{h}:00' for h in optimal_hours[:3]])} IST

Happy selling! 🛍️"""

        return RecommendationResponse(
            headline=headline,
            optimal_hours=optimal_hours[:3],
            products=products,
            whatsapp_message=whatsapp_message,
            persona_info=persona
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


# ---------- Static assets mount under /static ----------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
