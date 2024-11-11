# Stakeholder Auditing System for Non Profit Organization

This repository contains the implementation of a sophisticated stakeholder analysis tool designed for non-profit organizations (NPOs). It automatically crawls organization websites, identifies key stakeholders, and provides comprehensive analysis that aims to show organizations how different players perceive them in the nonprofit sector.

## Features

- 🔍 **Intelligent Web Crawling**
  - Automatic content extraction from organization websites
  - Respects domain boundaries and web crawling conventions
  - Smart rate limiting and timeout handling

- 🎯 **Stakeholder Identification**
  - Pre-defined profiles for common NPO stakeholders
  - Semantic similarity matching using OpenAI embeddings
  - Automatic relevance scoring and filtering

- 📊 **Comprehensive Analysis**
  - Detailed stakeholder impression analysis
  - Content quality evaluation
  - Relationship strength assessment
  - Sentiment analysis

- 📈 **Evaluation**
  - Coverage metrics
  - Content relevance scoring
  - Quality assessment
  - Sentiment distribution analysis
  - Detailed recommendations

## Workflow
![workflow_graph](https://github.com/user-attachments/assets/0ac0321f-9fb9-4439-8d24-ea1418ea4561)

## How It Works

The tool employs a sophisticated pipeline that combines web crawling, natural language processing, and AI analysis:

1. **Web Crawling and Content Processing**
   - Systematically crawls the organization's website while respecting domain boundaries
   - Extracts and cleans textual content from each webpage
   - Splits content into manageable chunks for processing

2. **Semantic Analysis and Storage**
   - Creates embeddings of content chunks using OpenAI's embedding model
   - Stores embeddings in a Chroma vector database for efficient semantic search
   - Enables retrieval of contextually relevant content using cosine similarity

3. **Stakeholder Identification**
   - For each potential stakeholder type, creates targeted search queries
   - Retrieves relevant content chunks using semantic similarity matching
   - Enriches prompts with stakeholder-specific parameters and interests
   - Uses LLM analysis to determine stakeholder relevance based on specific criteria
   - Filters results using configurable relevance thresholds

4. **Comprehensive Analysis via LLM**
   - Performs detailed analysis of how each identified stakeholder is served
   - Evaluates content relevance, clarity, and effectiveness
   - Generates stakeholder-specific impressions and recommendations

5. **Evaluation and Reporting**
   - Calculates coverage metrics across all potential stakeholders
   - Evaluates content quality and relevance scores
   - Performs sentiment analysis on stakeholder impressions
   - Generates detailed reports with actionable insights
   - Provides actionable improvement recommendations.
   - Scoring: Provides a 1-5 rating on how well the website serves each stakeholder's needs

## Prerequisites

- Python 3.11+
- OpenAI API key in `.env` file (OPENAI_API_KEY='sk-pr..............yoA')
- Required Python packages (see `requirements.txt`)

## Usage

1. Set the target website URL in `settings.py`
2. Run `npo_audit_engin.py`

## Project Structure

- `npo_audit_engine.py`: Main analysis orchestration
- `website_crawler.py`: Web crawling and content extraction
- `stakeholder_classes.py`: Stakeholder profile definitions
- `settings.py`: Configuration settings
- `logger_config.py`: Logging configuration
- `config.py`: Core data models and state management
- `eval.py`: Analysis evaluation system

## Output

The analysis generates several output files in the `analysis_output` directory:

- `final_audit.txt`: Detailed stakeholder audit results
- `stakeholder_analysis_info.txt`: Technical analysis information
- `evaluation_report.txt`: Comprehensive evaluation metrics
- `evaluation_metrics.json`: Machine-readable evaluation data
- `workflow_graph.png`: Visual representation of the analysis workflow

