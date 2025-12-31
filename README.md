# Meesho Micro-Moment Prediction Engine

## Demo Link
[https://drive.google.com/file/d/1-atFlcVSpobays6buo0IXeOJA9aQmuBv/view?usp=drive_link] (https://drive.google.com/file/d/1-atFlcVSpobays6buo0IXeOJA9aQmuBv/view?usp=drive_link)

# Installation & Setup:

**Clone Repository:-**

1) git clone https://github.com/aryan0xo76/meesho-micro-moments-predictor

2) cd meesho-micro-moments-predictor

**Create virtual environment:-**

python -m venv venv

**Activate virtual environment:-**

**a) On Windows:**

venv\Scripts\activate

**b) On macOS/Linux:**

source venv/bin/activate

**Install Dependencies:-**

pip install -r requirements.txt

**Run the Web Application Locally:-**

uvicorn main:app --reload --host 127.0.0.1 --port 8000

**The application will be available at: http://localhost:8000**

# Built for the DICE Challenge Season 2

An AI-powered WhatsApp marketing optimization platform that transforms basic seller inputs into personalized, high-converting WhatsApp campaigns.
This solution addresses the critical gap in WhatsApp marketing effectiveness for small sellers in India's social commerce ecosystem.

# What It Does

Input: Basic seller parameters (target users, product catalog, campaign duration)

Processing: Four ML models working together for optimization

Output: Ready-to-send WhatsApp marketing campaigns with optimal timing

Key Features
1) Personalized Product Recommendations - Co-visitation intelligence using PMI collaborative filtering

2) Optimal Send-Time Prediction - Empirical Bayes modeling for maximum engagement

3) Smart Product Re-ranking - Context-aware prioritization using gradient boosting

4) Dynamic Content Generation - Festival-aware headlines and WhatsApp message formatting
