"""
Headline Generator using Persona-specific Templates  
Generates contextually relevant WhatsApp message headlines
"""

import os
import random
import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import json

class HeadlineGenerator:
    """Persona-aware headline generator for WhatsApp messages"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trained = False
        
        # Persona-specific templates with relevant categories
        self.persona_templates = {
            "tier2_fashion": [
                "✨ Diwali Fashion Collection: Ethnic Wear Starting ₹299! ✨",
                "🌟 Festive Fashion: Designer Sarees & Kurtis Collection! 🌟", 
                "💃 Wedding Season Special: Ethnic Wear Up to 50% Off! 💃"
            ],
            "student_examprep": [
                "📚 Exam Essentials: Study Materials Up to 60% Off! 📚",
                "🎯 Student Special: Stationery & Electronics Bundle! 🎯",
                "⭐ Board Exam Ready: Complete Study Kit at Best Prices! ⭐"
            ],
            "budget_gadget": [
                "📱 Tech Flash Sale: Budget Electronics & Accessories! 📱",
                "🔋 Weekend Special: Phone Accessories Starting ₹99! 🔋",
                "🎧 Budget Tech Deals: Quality Gadgets, Pocket Prices! 🎧"
            ],
            "home_decor_festive": [
                "🏮 Diwali Home Decor: Transform Your Space for Festival! 🏮",
                "✨ Festive Home Collection: LED Lights & Diya Sets! ✨",
                "🎊 Pre-Diwali Sale: Home Decoration Items Up to 40% Off! 🎊"
            ],
            "regional_festive": [
                "🎭 Durga Puja Special: Traditional Bengali Collection! 🎭",
                "🙏 Festive Saree Collection: Pure Bengal Cotton & Silk! 🙏",
                "🎉 Regional Special: Authentic Puja Items & Ethnic Wear! 🎉"
            ]
        }
    
    def _get_occasion_from_calendar(self, df_calendar: pd.DataFrame) -> str:
        """Determine current occasion from calendar"""
        if df_calendar is None or df_calendar.empty:
            return "general"
        
        today = datetime.now().date().isoformat()
        active_events = df_calendar[df_calendar["date"] == today]
        
        if not active_events.empty:
            event_name = active_events.iloc[0]["festival_name"].lower()
            
            if "diwali" in event_name:
                return "diwali"
            elif "navratri" in event_name:
                return "navratri"
            elif "wedding" in event_name:
                return "wedding"
            elif "exam" in event_name:
                return "exam"
        
        return "general"
    
    async def train(self, df_products: pd.DataFrame, df_calendar: pd.DataFrame):
        """Train/initialize the headline generator"""
        print("Initializing headline generator...")
        self.trained = True
        await self.save_model()
        print("Headline generator ready")
    
    async def generate_headline(self, persona: Dict, df_calendar: pd.DataFrame = None) -> str:
        """Generate a contextually relevant headline - FIXED VERSION"""
        
        persona_id = persona.get("id", "general")
        
        # Get persona-specific headline
        if persona_id in self.persona_templates:
            headline = random.choice(self.persona_templates[persona_id])
        else:
            # Fallback to general template
            headline = "🎯 Special Picks: Premium Products Just for You! 🎯"
        
        return headline
    
    async def save_model(self):
        """Save model state"""
        os.makedirs("data", exist_ok=True)
        model_info = {
            "trained": self.trained,
            "device": "cpu",
            "transformers_available": False
        }
        
        with open("data/headlines_model_info.json", "w") as f:
            json.dump(model_info, f)
    
    async def load_model(self) -> bool:
        """Load saved model"""
        try:
            if os.path.exists("data/headlines_model_info.json"):
                with open("data/headlines_model_info.json", "r") as f:
                    model_info = json.load(f)
                
                self.trained = model_info.get("trained", False)
                return self.trained
        except Exception as e:
            print(f"Failed to load headline model: {e}")
        
        return False
