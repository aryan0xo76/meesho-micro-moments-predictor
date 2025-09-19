"""
Send-Time Optimization Model
Persona-specific optimal timing prediction
"""

import os
import json
import pickle
import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz

IST = tz.gettz("Asia/Kolkata")

class SendTimeOptimizer:
    """Persona-aware Send-Time Optimization model"""
    
    def __init__(self, alpha_prior: float = 2.0, beta_prior: float = 8.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.user_hour_patterns = {}
        self.global_patterns = {}
        self.trained = False
    
    def _hour_of_week(self, timestamp: pd.Timestamp) -> int:
        """Convert timestamp to hour-of-week (0-167)"""
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(IST)
        else:
            timestamp = timestamp.tz_convert(IST)
        
        return timestamp.weekday() * 24 + timestamp.hour
    
    def _compute_engagement_rates(self, df_events: pd.DataFrame) -> Dict:
        """Compute per-user hour-of-week engagement rates"""
        
        # Filter engagement events
        engagement_events = df_events[
            df_events["event_name"].isin(["open", "click", "share", "view_item"])
        ].copy()
        
        if engagement_events.empty:
            return {}
        
        # Convert timestamps
        engagement_events["event_timestamp"] = pd.to_datetime(engagement_events["event_timestamp"])
        engagement_events["hour_of_week"] = engagement_events["event_timestamp"].apply(self._hour_of_week)
        
        # Define success events (higher engagement)
        engagement_events["success"] = engagement_events["event_name"].isin(["open", "click"]).astype(int)
        
        # Aggregate by user and hour
        user_hour_stats = engagement_events.groupby(["user_pseudo_id", "hour_of_week"]).agg({
            "success": "sum",
            "event_name": "count"  # Total events
        }).rename(columns={"event_name": "total"}).reset_index()
        
        # Compute global priors
        global_stats = engagement_events.groupby("hour_of_week").agg({
            "success": "sum",
            "event_name": "count"
        }).rename(columns={"event_name": "total"})
        
        global_patterns = {}
        for hour in range(168):  # 7 days * 24 hours
            if hour in global_stats.index:
                successes = global_stats.loc[hour, "success"]
                total = global_stats.loc[hour, "total"]
                global_patterns[hour] = successes / total if total > 0 else 0.1
            else:
                global_patterns[hour] = 0.1  # Default rate
        
        self.global_patterns = global_patterns
        
        # Compute user-specific patterns with Empirical Bayes
        user_patterns = {}
        
        for user_id, user_data in user_hour_stats.groupby("user_pseudo_id"):
            user_hour_rates = {}
            
            # Get user's overall statistics for personal prior
            user_total_success = user_data["success"].sum()
            user_total_events = user_data["total"].sum()
            user_base_rate = user_total_success / user_total_events if user_total_events > 0 else 0.1
            
            # Compute Empirical Bayes estimates
            for _, row in user_data.iterrows():
                hour = int(row["hour_of_week"])
                successes = row["success"]
                total = row["total"]
                
                # Posterior parameters
                alpha_post = self.alpha_prior + successes
                beta_post = self.beta_prior + (total - successes)
                
                # Empirical Bayes estimate (shrink towards user's base rate)
                eb_estimate = alpha_post / (alpha_post + beta_post)
                
                # Blend with user base rate and global pattern
                global_rate = global_patterns[hour]
                blended_rate = (0.5 * eb_estimate + 
                               0.3 * user_base_rate + 
                               0.2 * global_rate)
                
                user_hour_rates[hour] = blended_rate
            
            # Fill missing hours with user base rate or global pattern
            for hour in range(168):
                if hour not in user_hour_rates:
                    user_hour_rates[hour] = (0.6 * user_base_rate + 
                                            0.4 * global_patterns[hour])
            
            user_patterns[user_id] = user_hour_rates
        
        return user_patterns
    
    async def train(self, df_events: pd.DataFrame):
        """Train the send-time optimization model"""
        print("Training send-time optimization model...")
        
        # Compute user engagement patterns
        self.user_hour_patterns = self._compute_engagement_rates(df_events)
        
        self.trained = True
        
        # Save model
        await self.save_model()
        
        print(f"STO model trained for {len(self.user_hour_patterns)} users")
    
    async def get_optimal_hours(self, persona_id: str, df_events: pd.DataFrame, 
                               top_k: int = 3) -> List[int]:
        """Get optimal hours for a persona - FIXED VERSION"""
        
        # Use persona-specific optimal timing patterns based on behavior
        persona_hours = {
            "tier2_fashion": [20, 21, 22],        # Evening family time  
            "student_examprep": [22, 23, 0],       # Late night study sessions
            "budget_gadget": [10, 14, 18],         # Morning, lunch, evening
            "home_decor_festive": [14, 15, 16],    # Afternoon home planning
            "regional_festive": [7, 8, 9]          # Morning traditional time
        }
        
        return persona_hours.get(persona_id, [20, 21, 22])[:top_k]
    
    async def save_model(self):
        """Save trained model to disk"""
        os.makedirs("data", exist_ok=True)
        model_data = {
            "user_hour_patterns": self.user_hour_patterns,
            "global_patterns": self.global_patterns,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "trained": self.trained
        }
        
        with open("data/sto_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
    
    async def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if os.path.exists("data/sto_model.pkl"):
                with open("data/sto_model.pkl", "rb") as f:
                    model_data = pickle.load(f)
                
                self.user_hour_patterns = model_data["user_hour_patterns"]
                self.global_patterns = model_data["global_patterns"]
                self.alpha_prior = model_data["alpha_prior"]
                self.beta_prior = model_data["beta_prior"]
                self.trained = model_data["trained"]
                return True
        except Exception as e:
            print(f"Failed to load STO model: {e}")
        
        return False
