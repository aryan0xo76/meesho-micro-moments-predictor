"""
Co-visitation Model for Product Recommendations
Session-based collaborative filtering with persona-aware category filtering
"""

import os
import pickle
import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

IST = tz.gettz("Asia/Kolkata")

class CovisitationModel:
    """Co-visitation based recommendation model with persona awareness"""
    
    def __init__(self):
        self.covisit_matrix = {}
        self.item_popularity = {}
        self.trained = False
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    def _sessionize_events(self, df_events: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
        """Group events into sessions with time-based gaps"""
        df = df_events.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        df = df.sort_values(["user_pseudo_id", "event_timestamp"])
        
        sessions = []
        for user_id, user_events in df.groupby("user_pseudo_id"):
            user_events = user_events.reset_index(drop=True)
            session_id = 0
            last_time = None
            
            for idx, row in user_events.iterrows():
                current_time = row["event_timestamp"]
                
                if last_time is None or (current_time - last_time).total_seconds() > gap_minutes * 60:
                    session_id += 1
                
                sessions.append({
                    **row.to_dict(),
                    "session_id_new": f"{user_id}_s{session_id}"
                })
                last_time = current_time
        
        return pd.DataFrame(sessions)
    
    def _build_covisitation_matrix(self, df_sessions: pd.DataFrame) -> Dict:
        """Build item-item co-visitation matrix with action weighting"""
        
        # Action weights
        action_weights = {
            "purchase": 5.0,
            "add_to_cart": 3.0,
            "click": 2.0,
            "view_item": 1.0,
            "view_item_list": 0.5,
            "share": 1.5,
            "open": 1.0
        }
        
        # Filter relevant events
        relevant_events = df_sessions[
            df_sessions["event_name"].isin(action_weights.keys())
        ].copy()
        
        if relevant_events.empty:
            return {}
        
        # Add weights
        relevant_events["weight"] = relevant_events["event_name"].map(action_weights).fillna(1.0)
        
        covisit_pairs = {}
        item_counts = {}
        
        # Process each session
        for session_id, session_events in relevant_events.groupby("session_id_new"):
            items = list(session_events[["product_id", "weight", "event_timestamp"]].itertuples(index=False, name=None))
            
            # Count individual items
            for item_id, weight, timestamp in items:
                item_counts[item_id] = item_counts.get(item_id, 0) + weight
            
            # Create pairs within session
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item1, weight1, time1 = items[i]
                    item2, weight2, time2 = items[j]
                    
                    if item1 == item2:
                        continue
                    
                    # Time decay (items closer in time have higher weight)
                    time_diff_minutes = abs((time2 - time1).total_seconds()) / 60.0
                    time_decay = 1.0 / (1.0 + 0.01 * time_diff_minutes)
                    
                    # Combined weight
                    pair_weight = (weight1 + weight2) * time_decay
                    
                    # Store both directions
                    pair_key1 = (item1, item2)
                    pair_key2 = (item2, item1)
                    
                    covisit_pairs[pair_key1] = covisit_pairs.get(pair_key1, 0) + pair_weight
                    covisit_pairs[pair_key2] = covisit_pairs.get(pair_key2, 0) + pair_weight
        
        # Normalize by item popularity
        normalized_covisits = {}
        for (item1, item2), weight in covisit_pairs.items():
            # Normalize by geometric mean of individual item popularity
            norm_factor = np.sqrt(item_counts.get(item1, 1) * item_counts.get(item2, 1))
            normalized_weight = weight / norm_factor
            normalized_covisits[(item1, item2)] = normalized_weight
        
        self.item_popularity = item_counts
        return normalized_covisits
    
    async def train(self, df_events: pd.DataFrame):
        """Train the co-visitation model"""
        print("Training co-visitation model...")
        
        # Sessionize events
        df_sessions = self._sessionize_events(df_events)
        
        # Build co-visitation matrix
        self.covisit_matrix = self._build_covisitation_matrix(df_sessions)
        
        self.trained = True
        
        # Save model
        await self.save_model()
        
        print(f"Co-visitation model trained with {len(self.covisit_matrix)} pairs")
    
    async def get_recommendations(self, preferred_categories: List[str], 
                                  df_products: pd.DataFrame, limit: int = 10) -> List[Dict]:
        """Get recommendations based on category preferences - FIXED VERSION"""
        
        # ALWAYS filter by preferred categories first
        category_products = df_products[
            df_products["category"].isin(preferred_categories)
        ]
        
        if category_products.empty:
            # If no products in preferred categories, use all products as fallback
            category_products = df_products
        
        # Sort by product_id for consistent ordering
        recommended = category_products.sort_values("product_id").head(limit)
        
        recommendations = []
        for _, row in recommended.iterrows():
            recommendations.append({
                "product_id": int(row["product_id"]),
                "title": row["title"],
                "category": row["category"],
                "price": float(row["price"]),
                "score": 1.0,
                "reason": f"Matches {row['category']} preference"
            })
        
        return recommendations
    
    async def save_model(self):
        """Save trained model to disk"""
        os.makedirs("data", exist_ok=True)
        model_data = {
            "covisit_matrix": self.covisit_matrix,
            "item_popularity": self.item_popularity,
            "trained": self.trained
        }
        
        with open("data/covisitation_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
    
    async def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if os.path.exists("data/covisitation_model.pkl"):
                with open("data/covisitation_model.pkl", "rb") as f:
                    model_data = pickle.load(f)
                
                self.covisit_matrix = model_data["covisit_matrix"]
                self.item_popularity = model_data["item_popularity"]
                self.trained = model_data["trained"]
                return True
        except Exception as e:
            print(f"Failed to load co-visitation model: {e}")
        
        return False
