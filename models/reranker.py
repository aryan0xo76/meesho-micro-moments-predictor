"""
CatBoost GPU Re-ranker Model
CUDA 11.8 compatible gradient boosting for recommendation re-ranking
"""

import os
import pickle
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Try to import CatBoost with GPU support
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class CatBoostReranker:
    """CatBoost-based re-ranking model with GPU support"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.trained = False
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available() and CATBOOST_AVAILABLE
        self.device = "GPU" if self.use_gpu else "CPU"
    
    def _extract_features(self, df_events: pd.DataFrame, df_products: pd.DataFrame) -> pd.DataFrame:
        """Extract features for training the re-ranker"""
        
        # Create training samples from user sessions
        training_samples = []
        
        # Sort events by user and time
        df_events = df_events.sort_values(["user_pseudo_id", "event_timestamp"])
        df_events["event_timestamp"] = pd.to_datetime(df_events["event_timestamp"])
        
        # Group by user sessions
        for user_id, user_events in df_events.groupby("user_pseudo_id"):
            user_events = user_events.reset_index(drop=True)
            
            # Create simple sessions (gap-based)
            sessions = []
            current_session = []
            last_time = None
            
            for _, event in user_events.iterrows():
                current_time = event["event_timestamp"]
                
                if last_time is None or (current_time - last_time).total_seconds() > 1800:  # 30 min gap
                    if current_session:
                        sessions.append(current_session)
                    current_session = [event]
                else:
                    current_session.append(event)
                
                last_time = current_time
            
            if current_session:
                sessions.append(current_session)
            
            # Generate training samples from sessions
            for session in sessions:
                if len(session) < 2:
                    continue
                
                # Use last item as target, previous items as context
                target_event = session[-1]
                context_events = session[:-1]
                
                if target_event["event_name"] not in ["view_item", "add_to_cart", "purchase"]:
                    continue
                
                target_product_id = target_event["product_id"]
                target_category = target_event["category"]
                
                # Get context features
                context_categories = [e["category"] for e in context_events]
                context_products = [e["product_id"] for e in context_events]
                
                # Get product info for target
                target_product = df_products[df_products["product_id"] == target_product_id]
                if target_product.empty:
                    continue
                
                target_price = target_product.iloc[0]["price"]
                
                # Generate candidate features (positive sample)
                features = self._compute_item_features(
                    target_product_id, target_category, target_price,
                    context_products, context_categories, df_products
                )
                features["label"] = 1  # Positive sample
                training_samples.append(features)
                
                # Generate negative samples (limited to avoid memory issues)
                negative_products = df_products[
                    ~df_products["product_id"].isin(context_products + [target_product_id])
                ].sample(n=min(2, len(df_products)), random_state=42)
                
                for _, neg_product in negative_products.iterrows():
                    neg_features = self._compute_item_features(
                        neg_product["product_id"], neg_product["category"], neg_product["price"],
                        context_products, context_categories, df_products
                    )
                    neg_features["label"] = 0  # Negative sample
                    training_samples.append(neg_features)
        
        if not training_samples:
            return pd.DataFrame()
        
        return pd.DataFrame(training_samples)
    
    def _compute_item_features(self, item_id: int, item_category: str, item_price: float,
                              context_products: List[int], context_categories: List[str],
                              df_products: pd.DataFrame) -> Dict:
        """Compute features for an item given context"""
        
        # Category match with context
        category_match = 1 if item_category in context_categories else 0
        category_match_ratio = sum(1 for c in context_categories if c == item_category) / max(len(context_categories), 1)
        
        # Price features
        context_prices = []
        for pid in context_products:
            product_info = df_products[df_products["product_id"] == pid]
            if not product_info.empty:
                context_prices.append(product_info.iloc[0]["price"])
        
        if context_prices:
            avg_context_price = np.mean(context_prices)
            price_diff = item_price - avg_context_price
            price_ratio = item_price / avg_context_price if avg_context_price > 0 else 1.0
        else:
            price_diff = 0.0
            price_ratio = 1.0
        
        # Item popularity (frequency in dataset)
        item_popularity = len(df_products[df_products["category"] == item_category])
        
        return {
            "category_match": category_match,
            "category_match_ratio": category_match_ratio,
            "price_diff": price_diff,
            "price_ratio": price_ratio,
            "item_popularity": item_popularity,
            "item_price": item_price,
            "context_size": len(context_products)
        }
    
    async def train(self, df_events: pd.DataFrame, df_products: pd.DataFrame):
        """Train the CatBoost re-ranker"""
        print("Training CatBoost re-ranker...")
        
        if not CATBOOST_AVAILABLE:
            print("CatBoost not available. Skipping re-ranker training.")
            return
        
        # Extract features
        training_data = self._extract_features(df_events, df_products)
        
        if training_data.empty:
            print("No training data generated. Skipping re-ranker training.")
            return
        
        # Prepare features and labels
        feature_cols = [col for col in training_data.columns if col != "label"]
        X = training_data[feature_cols]
        y = training_data["label"]
        
        if len(X) < 10:
            print("Insufficient training data for CatBoost. Skipping.")
            return
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create CatBoost model
        self.model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=4,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type=self.device,
            verbose=0,
            random_seed=42
        )
        
        # Train model
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            # Evaluate
            val_pred = self.model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, val_pred)
            
            print(f"Re-ranker trained. Validation AUC: {auc_score:.3f}")
            print(f"Device used: {self.device}")
            
            self.trained = True
            
            # Save model
            await self.save_model()
            
        except Exception as e:
            print(f"CatBoost training failed: {e}")
            print("Continuing without re-ranker...")
    
    async def save_model(self):
        """Save trained model to disk"""
        if self.model and CATBOOST_AVAILABLE:
            self.model.save_model("data/reranker.cbm")
            
            # Save metadata
            metadata = {
                "feature_names": self.feature_names,
                "trained": self.trained,
                "use_gpu": self.use_gpu
            }
            
            with open("data/reranker_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
    
    async def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if CATBOOST_AVAILABLE and os.path.exists("data/reranker.cbm"):
                self.model = CatBoostClassifier()
                self.model.load_model("data/reranker.cbm")
                
                # Load metadata
                if os.path.exists("data/reranker_metadata.pkl"):
                    with open("data/reranker_metadata.pkl", "rb") as f:
                        metadata = pickle.load(f)
                    
                    self.feature_names = metadata["feature_names"]
                    self.trained = metadata["trained"]
                    self.use_gpu = metadata.get("use_gpu", False)
                
                return True
        except Exception as e:
            print(f"Failed to load re-ranker model: {e}")
        
        return False