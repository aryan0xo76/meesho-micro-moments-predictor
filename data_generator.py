"""
Data Generator for GA4/GSC-style synthetic data
Creates realistic Indian e-commerce datasets
"""

import os
import random
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dateutil import tz

# IST Timezone
IST = tz.gettz("Asia/Kolkata")

class DataGenerator:
    """Generate synthetic data that matches GA4/GSC export schemas"""
    
    def __init__(self):
        self.personas = [
            {
                "id": "tier2_fashion",
                "preferred_categories": ["fashion_ethnic", "jewelry", "accessories", "kurta", "saree"],
                "peak_hours": [20, 21, 22],
                "boost_events": ["Diwali", "Wedding Season", "Navratri"]
            },
            {
                "id": "student_examprep", 
                "preferred_categories": ["stationery", "electronics_budget", "books"],
                "peak_hours": [22, 23, 0, 1],
                "boost_events": ["Board Exams", "Competitive Exams"]
            },
            {
                "id": "budget_gadget",
                "preferred_categories": ["electronics_accessories", "gadgets_budget", "mobile_accessories"],
                "peak_hours": [18, 19, 20, 21],
                "boost_events": ["Diwali Sale", "Republic Day Sale"]
            },
            {
                "id": "home_decor_festive",
                "preferred_categories": ["home_decor", "lighting", "furnishing"],
                "peak_hours": [14, 15, 16, 17],
                "boost_events": ["Pre-Diwali", "Gudi Padwa", "House Warming"]
            },
            {
                "id": "regional_festive",
                "preferred_categories": ["saree", "ethnic_bengali", "puja_items"],
                "peak_hours": [7, 8, 9, 10],
                "boost_events": ["Durga Puja", "Kali Puja", "Poila Boishakh"]
            }
        ]
        
        self.categories = [
            "fashion_ethnic", "jewelry", "accessories", "kurta", "saree", "lehenga",
            "ethnic_bengali", "puja_items", "stationery", "books", "electronics_budget",
            "electronics_accessories", "gadgets_budget", "mobile_accessories", 
            "home_decor", "lighting", "furnishing", "designer_wear", "footwear", "bags"
        ]
        
        self.geo_states = ["MH", "DL", "WB", "KA", "TN", "GJ", "RJ", "UP", "MP", "KL", "HR", "PB", "AS", "BR", "OD"]
        
    async def generate_products(self, n_products: int, seed: int = 42) -> str:
        """Generate products.csv with Indian product names"""
        rng = np.random.default_rng(seed)
        
        products = []
        product_names = {
            "fashion_ethnic": ["Cotton Kurta Set", "Ethnic Kurti Collection", "Traditional Dupatta"],
            "jewelry": ["Gold Plated Necklace Set", "Silver Jhumka Earrings", "Kundan Jewelry"],
            "accessories": ["Designer Handbag", "Ethnic Clutch", "Traditional Belt"],
            "kurta": ["Cotton Kurta", "Silk Kurta", "Designer Kurta Set", "Festive Kurta"],
            "saree": ["Cotton Saree", "Silk Wedding Saree", "Georgette Saree", "Handloom Saree"],
            "lehenga": ["Designer Lehenga", "Bridal Lehenga", "Festive Lehenga Set"],
            "ethnic_bengali": ["Bengali Cotton Saree", "Tant Saree", "Bengali Silk Saree", "Dhaka Jamdani"],
            "puja_items": ["Brass Puja Thali", "Silver Kalash", "Puja Diya Set", "Incense Holder"],
            "stationery": ["Complete Stationery Kit", "Scientific Calculator", "Geometry Box Set"],
            "books": ["Study Guide", "Reference Book", "Exam Prep Book"],
            "electronics_budget": ["Budget Earbuds", "Phone Charger", "Power Bank"],
            "electronics_accessories": ["Phone Case", "Screen Protector", "Cable Organizer"],
            "gadgets_budget": ["Bluetooth Speaker", "Wireless Mouse", "USB Hub"],
            "mobile_accessories": ["Phone Stand", "Car Charger", "Selfie Stick"],
            "home_decor": ["LED Diwali Lights", "Decorative Rangoli Stencil", "Brass Diya Set"],
            "lighting": ["String Lights", "Decorative Lamp", "LED Candles"],
            "furnishing": ["Cushion Cover", "Table Runner", "Wall Hanging"]
        }
        
        for pid in range(1, n_products + 1):
            # Weight categories for Indian market - ensure persona categories get proper weight
            category_weights = [0.12, 0.08, 0.06, 0.08, 0.12, 0.06, 0.08, 0.06, 0.08, 0.06, 0.08, 0.06, 0.06]
            category_weights = np.array(category_weights[:len(self.categories)])
            category_weights = category_weights / category_weights.sum()
            cat = rng.choice(self.categories[:len(category_weights)], p=category_weights)

            
            # Generate realistic names
            if cat in product_names:
                name = rng.choice(product_names[cat])
            else:
                name = f"{cat.replace('_', ' ').title()} Item"
            
            # Price based on category
            if "electronics" in cat:
                price = float(np.clip(rng.normal(899, 300), 199, 2999))
            elif cat in ["saree", "kurta", "lehenga"]:
                price = float(np.clip(rng.normal(599, 200), 299, 1999))
            else:
                price = float(np.clip(rng.normal(399, 150), 99, 999))
            
            products.append([
                pid,
                f"{name} {pid}",
                f"High quality {cat.replace('_', ' ')} product for Indian customers",
                cat,
                price,
                int(rng.integers(1, 6))
            ])
        
        df = pd.DataFrame(products, columns=[
            "product_id", "title", "description", "category", "price", "image_count"
        ])
        
        filepath = "data/products.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    async def generate_users(self, n_users: int, seed: int = 42) -> str:
        """Generate users.csv with persona assignments"""
        rng = np.random.default_rng(seed)
        
        reseller_ids = [f"R{i:03d}" for i in range(1, 101)]
        users = []
        
        for uid in range(1, n_users + 1):
            persona = rng.choice(self.personas)
            
            users.append([
                f"U{uid:06d}",
                rng.choice(reseller_ids),
                "IST",
                rng.choice(self.geo_states),
                (datetime.now(tz=IST) - timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d"),
                persona["id"]
            ])
        
        df = pd.DataFrame(users, columns=[
            "user_pseudo_id", "reseller_id", "timezone", "geo_state", "created_at", "persona_id"
        ])
        
        filepath = "data/users.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    async def generate_calendar(self, days: int, seed: int = 42) -> str:
        """Generate calendar.csv with Indian festivals"""
        start_date = datetime.now(tz=IST) - timedelta(days=days)
        end_date = datetime.now(tz=IST)
        
        festivals = [
            {"name": "Diwali", "region": "pan_india", "start_offset": -30, "duration": 10, "weight": 2.5},
            {"name": "Navratri", "region": "west_india", "start_offset": -45, "duration": 9, "weight": 2.0},
            {"name": "Durga Puja", "region": "east_india", "start_offset": -35, "duration": 7, "weight": 2.2},
            {"name": "Wedding Season", "region": "pan_india", "start_offset": -10, "duration": 90, "weight": 1.8},
            {"name": "Board Exams", "region": "pan_india", "start_offset": -60, "duration": 45, "weight": 1.5},
            {"name": "Onam", "region": "south_india", "start_offset": -80, "duration": 5, "weight": 1.7},
            {"name": "Karva Chauth", "region": "north_india", "start_offset": -25, "duration": 1, "weight": 1.6}
        ]
        
        cal_rows = []
        for festival in festivals:
            festival_start = end_date + timedelta(days=festival["start_offset"])
            festival_end = festival_start + timedelta(days=festival["duration"])
            
            current = max(festival_start, start_date)
            while current <= min(festival_end, end_date):
                cal_rows.append([
                    current.date().isoformat(),
                    festival["name"],
                    festival["region"],
                    festival["weight"]
                ])
                current += timedelta(days=1)
        
        df = pd.DataFrame(cal_rows, columns=["date", "festival_name", "region", "season_weight"])
        
        filepath = "data/calendar.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    async def generate_gsc_queries(self, days: int, seed: int = 42) -> str:
        """Generate gsc_queries.csv matching Search Console format"""
        rng = np.random.default_rng(seed)
        start_date = datetime.now(tz=IST) - timedelta(days=days)
        
        queries = [
            "diwali kurta for men", "women saree festive", "durga puja saree", 
            "rangoli stencil", "exam calculator", "geometry box", 
            "budget earbuds", "home lighting diwali", "navratri lehenga",
            "wedding jewelry set", "artificial jewelry", "cotton kurta price",
            "board exam stationery", "scientific calculator", "diya set diwali"
        ]
        
        rows = []
        current = start_date
        
        while current <= datetime.now(tz=IST):
            for query in queries:
                base_impressions = 50 + rng.integers(0, 200)
                
                # Seasonal boosts
                multiplier = 1.0
                if "diwali" in query.lower() and current.month in [10, 11]:
                    multiplier = 2.5
                elif "exam" in query.lower() and current.month in [2, 3, 4]:
                    multiplier = 1.8
                elif "wedding" in query.lower() and current.month in [11, 12, 1, 2]:
                    multiplier = 2.0
                
                impressions = int(base_impressions * multiplier)
                clicks = max(0, int(impressions * (0.03 + rng.random() * 0.07)))
                ctr = clicks / impressions if impressions > 0 else 0
                position = 2.0 + rng.random() * 6.0
                
                rows.append([
                    current.date().isoformat(),
                    query,
                    impressions,
                    clicks,
                    round(ctr, 4),
                    round(position, 2)
                ])
            
            current += timedelta(days=1)
        
        df = pd.DataFrame(rows, columns=["date", "query", "impressions", "clicks", "ctr", "position"])
        
        filepath = "data/gsc_queries.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    async def generate_events(self, n_users: int, n_products: int, days: int, seed: int = 42) -> str:
        """Generate events.csv matching GA4 export format"""
        rng = np.random.default_rng(seed)
        start_date = datetime.now(tz=IST) - timedelta(days=days)
        end_date = datetime.now(tz=IST)
        
        # Load users for persona info
        if os.path.exists("data/users.csv"):
            df_users = pd.read_csv("data/users.csv")
        else:
            # Fallback user generation
            df_users = pd.DataFrame({
                "user_pseudo_id": [f"U{i:06d}" for i in range(1, n_users+1)],
                "persona_id": [rng.choice([p["id"] for p in self.personas]) for _ in range(n_users)]
            })
        
        events = []
        event_types = ["view_item", "view_item_list", "add_to_cart", "purchase", "share", "open", "click", "view_search_results"]
        
        # Search terms by category
        search_terms = {
            "fashion_ethnic": ["diwali kurta", "cotton saree price", "ethnic wear"],
            "stationery": ["exam calculator", "geometry box", "blue pen"],
            "electronics_budget": ["budget earbuds price", "phone charger", "usb cable"],
            "home_decor": ["diwali lights", "rangoli stencil", "led string lights"]
        }
        
        for _, user_row in df_users.iterrows():
            uid = user_row["user_pseudo_id"]
            persona_id = user_row.get("persona_id", "tier2_fashion")
            
            # Find persona config
            persona = next((p for p in self.personas if p["id"] == persona_id), self.personas[0])
            
            # Generate sessions for this user
            num_sessions = int(rng.integers(3, 15))
            
            for session_idx in range(num_sessions):
                # Random session timestamp
                session_start = start_date + timedelta(
                    days=int(rng.integers(0, days)),
                    hours=int(rng.integers(0, 24)),
                    minutes=int(rng.integers(0, 60))
                )
                session_start = session_start.replace(tzinfo=IST)
                
                session_id = f"{uid}_{int(session_start.timestamp())}"
                
                # Session events
                num_events = int(rng.integers(2, 8))
                current_time = session_start
                
                for event_idx in range(num_events):
                    event_type = rng.choice(event_types, p=[0.3, 0.15, 0.15, 0.05, 0.05, 0.15, 0.1, 0.05])
                    
                    # Bias timing for engagement events
                    if event_type in ["open", "click", "share"]:
                        hour = current_time.hour
                        if rng.random() < 0.7 and hour not in persona["peak_hours"]:
                            # Shift to peak hour
                            peak_hour = rng.choice(persona["peak_hours"])
                            current_time = current_time.replace(hour=peak_hour)
                    
                    # Select product and category
                    if rng.random() < 0.6:
                        category = rng.choice(persona["preferred_categories"])
                    else:
                        category = rng.choice(self.categories)
                    
                    product_id = int(rng.integers(1, n_products + 1))
                    
                    # Generate search term for search events
                    search_term = ""
                    if event_type == "view_search_results":
                        if category in search_terms:
                            search_term = rng.choice(search_terms[category])
                        else:
                            search_term = f"{category.replace('_', ' ')}"
                    
                    # Randomly select reseller and geo_state
                    reseller_id = f"R{rng.integers(1, 101):03d}"
                    geo_state = rng.choice(self.geo_states)
                    
                    events.append([
                        uid,
                        reseller_id,
                        current_time.isoformat(),
                        session_id,
                        event_type,
                        product_id,
                        category,
                        "whatsapp",
                        search_term,
                        geo_state,
                        "mobile"
                    ])
                    
                    # Increment time
                    current_time += timedelta(minutes=int(rng.integers(1, 10)))
        
        df = pd.DataFrame(events, columns=[
            "user_pseudo_id", "reseller_id", "event_timestamp", "session_id", 
            "event_name", "product_id", "category", "channel", "search_term", 
            "geo_state", "device_type"
        ])
        
        filepath = "data/events.csv"
        df.to_csv(filepath, index=False)
        return filepath
    
    async def generate_all_data(self, n_users: int = 8000, n_products: int = 3000, 
                               days: int = 120, seed: int = 42) -> List[str]:
        """Generate all data files"""
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        files_created = []
        
        # Generate in order (some depend on others)
        files_created.append(await self.generate_products(n_products, seed))
        files_created.append(await self.generate_users(n_users, seed))
        files_created.append(await self.generate_calendar(days, seed))
        files_created.append(await self.generate_gsc_queries(days, seed))
        files_created.append(await self.generate_events(n_users, n_products, days, seed))
        
        return files_created