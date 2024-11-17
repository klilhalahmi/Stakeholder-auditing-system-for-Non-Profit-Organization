"""
Configuration settings for the stakeholder analysis application.

This module manages all the configuration settings including API keys, URLs,
model selections, and analysis parameters. It loads environment variables
and defines constants used throughout the application.

Environment Variables Required:
    OPENAI_API_KEY: OpenAI API key for accessing GPT models
"""

from dotenv import load_dotenv
import os
from stakeholder_classes import StakeholderRegistry

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API key for OpenAI services

# Target Website Configuration
WEBSITE_URL = "https://actsofwisdom.com/"  # Website to analyze for stakeholders

# Output Configuration
OUTPUT_DIR = "analysis_output"  # Directory for storing analysis results

# Model Selection
EVAL_MODEL = "gpt-4-turbo"      # Model used for evaluation tasks (higher accuracy)
MODEL = "gpt-3.5-turbo-16k"     # Model used for general analysis (better cost/performance)

# Stakeholder Configuration
# Get comprehensive list of potential stakeholders from registry
TOTAL_PLAYERS = StakeholderRegistry.get_all_stakeholders()

# Analysis Parameters
RELEVANCE_THRESHOLD = 0.65  # Minimum similarity score for stakeholder relevance (0.0 to 1.0)
MAX_RELEVANT = 7           # Maximum number of relevant stakeholders to display
