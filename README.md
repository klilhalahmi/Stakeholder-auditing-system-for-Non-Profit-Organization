# Stakeholder-auditing-system-for-Non-Profit-Organization

This repository contains the implementation of a sophisticated stakeholder analysis tool designed for non-profit organizations (NPOs). It automatically crawls organization websites, identifies key stakeholders, and provides comprehensive analysis of how well the organization serves different stakeholder needs.

## Workflow

![Stakeholder-auditing-system-for-Non-Profit-Organization](analysis_output/workflow_graph.png)

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

## Prerequisites

- Python 3.11+
- OpenAI API key in ".env" file (OPENAI_API_KEY='sk-proj--sW...yoA')
- Required Python packages (see `requirements.txt`)

## Usage

1. Set the target website URL in `settings.py`
2. Run `npo_audit_engin.py`

The tool will:
1. Crawl the specified website
2. Create embeddings of the content
3. Identify relevant stakeholders
4. Analyze stakeholder relationships
5. Generate comprehensive reports

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


[Add your chosen license here]

## Contact

[Add your contact information]
