"""
Stakeholder Analysis Evaluation System

This module provides comprehensive evaluation capabilities for stakeholder analysis results,
including metrics for coverage, relevance, content quality, and sentiment analysis.
"""

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from settings import OPENAI_API_KEY, EVAL_MODEL, TOTAL_PLAYERS
from logger_config import CustomLogger

logger = CustomLogger().get_logger("Evaluator")

@dataclass(slots=True)
class EvaluationMetrics:
    """
    Container for comprehensive stakeholder analysis evaluation metrics.
    
    Attributes:
        total_stakeholders (int): Total number of potential stakeholders
        identified_count (int): Number of stakeholders actually identified
        coverage_rate (float): Percentage of potential stakeholders identified
        avg_relevance_score (float): Mean relevance score across all content
        min_relevance_score (float): Lowest relevance score found
        max_relevance_score (float): Highest relevance score found
        avg_chunks_per_stakeholder (float): Average number of content chunks per stakeholder
        content_quality_score (float): Overall quality score for content (0-1)
        justification_quality_score (float): Quality score for identification justifications (0-1)
        impression_clarity_score (float): Clarity score for stakeholder impressions (0-1)
        sentiment_distribution (Dict[str, float]): Distribution of sentiment classifications
        completeness_score (float): Overall completeness of analysis (0-1)
        reliability_score (float): Overall reliability of analysis (0-1)
        evaluation_timestamp (str): When the evaluation was performed
    """
    total_stakeholders: int
    identified_count: int
    coverage_rate: float
    avg_relevance_score: float
    min_relevance_score: float
    max_relevance_score: float
    avg_chunks_per_stakeholder: float
    content_quality_score: float
    justification_quality_score: float
    impression_clarity_score: float
    sentiment_distribution: Dict[str, float]
    completeness_score: float
    reliability_score: float
    evaluation_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class StakeholderAnalysisEvaluator:
    """
    Evaluator for assessing the quality and completeness of stakeholder analysis results.
    
    This class provides methods to evaluate various aspects of stakeholder analysis:
    - Coverage of identified stakeholders
    - Quality of content and justifications
    - Clarity of stakeholder impressions
    - Sentiment analysis of impressions
    - Overall completeness and reliability scores
    
    Args:
        output_dir (str): Directory where evaluation reports will be saved
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.eval_llm = ChatOpenAI(
            model=EVAL_MODEL,
            temperature=0.2,
            api_key=OPENAI_API_KEY
        )
    
    def analyze_sentiment(self, impression: str) -> str:
        """
        Analyze the sentiment of a stakeholder impression using LLM.
        
        Args:
            impression (str): Stakeholder impression text
            
        Returns:
            str: Sentiment classification (VERY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, VERY_NEGATIVE)
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Analyze the stakeholder impression and classify it as one of:
                VERY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, VERY_NEGATIVE
                Respond with just the classification."""),
            HumanMessage(content=f"Impression: {impression}")
        ])
        
        result = self.eval_llm.invoke(prompt.format_messages())
        return result.content.strip()
    
    def evaluate_justification_quality(self, justification: str) -> float:
        """
        Evaluate the quality of stakeholder identification justification.
        
        Assesses clarity, specificity, and logical connection of the justification.
        
        Args:
            justification (str): Justification text for stakeholder identification
            
        Returns:
            float: Quality score between 0 and 1
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Rate the quality of the stakeholder identification justification on a scale of 0-1:
                Consider:
                - Clarity of reasoning
                - Specificity of evidence
                - Logical connection
                Respond with just the numerical score."""),
            HumanMessage(content=f"Justification: {justification}")
        ])
        
        try:
            result = self.eval_llm.invoke(prompt.format_messages())
            return float(result.content.strip())
        except Exception as e:
            logger.error(f"Error evaluating justification: {str(e)}")
            return 0.0
    
    def evaluate_impression_clarity(self, impression: str) -> float:
        """
        Evaluate the clarity and specificity of stakeholder impression.
        
        Args:
            impression (str): Stakeholder impression text
            
        Returns:
            float: Clarity score between 0 and 1
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Rate the clarity and specificity of the stakeholder impression on a scale of 0-1:
                Consider:
                - Concreteness of observations
                - Specificity of relationship description
                - Clarity of sentiment
                Respond with just the numerical score."""),
            HumanMessage(content=f"Impression: {impression}")
        ])
        
        try:
            result = self.eval_llm.invoke(prompt.format_messages())
            return float(result.content.strip())
        except Exception as e:
            logger.error(f"Error evaluating impression clarity: {str(e)}")
            return 0.0
    
    def evaluate_content_quality(self, content_chunks: List[Dict]) -> float:
        """
        Evaluate the quality of content chunks for a stakeholder.
        
        Considers both relevance scores and content length.
        
        Args:
            content_chunks (List[Dict]): List of content chunks with scores
            
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            relevance_scores = [chunk['score'] for chunk in content_chunks]
            content_length = sum(len(chunk['content']) for chunk in content_chunks)
            
            return np.mean([
                np.mean(relevance_scores),
                min(1.0, content_length / 1000)
            ])
        except Exception as e:
            logger.error(f"Error evaluating content quality: {str(e)}")
            return 0.0


    def evaluate_results(self, state) -> EvaluationMetrics:
        """Perform comprehensive evaluation of stakeholder analysis results."""
        logger.info("Starting evaluation of stakeholder analysis results")
        
        try:
            # Coverage metrics calculation
            total_stakeholders = len(TOTAL_PLAYERS)
            identified_count = len(state.identified_stakeholders.identified_stakeholders)
            coverage_rate = identified_count / total_stakeholders
            
            # Relevance metrics calculation
            all_scores = []
            for stakeholder in state.identified_stakeholders.identified_stakeholders:
                chunks = state.identified_stakeholders.stakeholder_data[stakeholder.name]
                all_scores.extend([chunk['score'] for chunk in chunks])
                
            avg_relevance_score = np.mean(all_scores) if all_scores else 0
            min_relevance_score = min(all_scores) if all_scores else 0
            max_relevance_score = max(all_scores) if all_scores else 0
            
            # Content metrics calculation
            chunks_per_stakeholder = [
                len(state.identified_stakeholders.stakeholder_data[s.name])
                for s in state.identified_stakeholders.identified_stakeholders
            ]
            avg_chunks = np.mean(chunks_per_stakeholder) if chunks_per_stakeholder else 0
            
            # Initialize scoring lists
            content_scores = []
            justification_scores = []
            impression_scores = []
            sentiments = []
            
            # Evaluate each stakeholder
            for stakeholder in state.identified_stakeholders.identified_stakeholders:
                try:
                    # Content quality evaluation
                    content_score = self.evaluate_content_quality(
                        state.identified_stakeholders.stakeholder_data[stakeholder.name]
                    )
                    content_scores.append(content_score)
                    
                    # Justification quality evaluation
                    justification_score = self.evaluate_justification_quality(
                        state.identified_stakeholders.justification[stakeholder.name]
                    )
                    justification_scores.append(justification_score)
                    
                    # Impression clarity evaluation
                    impression = state.stakeholder_audit.stakeholder_impressions[stakeholder.name]
                    impression_score = self.evaluate_impression_clarity(impression)
                    impression_scores.append(impression_score)
                    
                    # Sentiment analysis
                    sentiment = self.analyze_sentiment(impression)
                    sentiments.append(sentiment)
                    
                except Exception as e:
                    logger.error(f"Error processing stakeholder {stakeholder.name}: {str(e)}")
                    continue
            
            # Calculate aggregate scores
            content_quality_score = np.mean(content_scores) if content_scores else 0
            justification_quality_score = np.mean(justification_scores) if justification_scores else 0
            impression_clarity_score = np.mean(impression_scores) if impression_scores else 0
            
            # Calculate sentiment distribution
            sentiment_counts = pd.Series(sentiments).value_counts()
            sentiment_distribution = sentiment_counts.to_dict()
            
            # Calculate overall scores
            completeness_score = np.mean([
                coverage_rate,
                content_quality_score,
                impression_clarity_score
            ])
            
            reliability_score = np.mean([
                avg_relevance_score,
                justification_quality_score,
                impression_clarity_score
            ])
            
            # Create and save metrics
            metrics = EvaluationMetrics(
                total_stakeholders=total_stakeholders,
                identified_count=identified_count,
                coverage_rate=coverage_rate,
                avg_relevance_score=avg_relevance_score,
                min_relevance_score=min_relevance_score,
                max_relevance_score=max_relevance_score,
                avg_chunks_per_stakeholder=avg_chunks,
                content_quality_score=content_quality_score,
                justification_quality_score=justification_quality_score,
                impression_clarity_score=impression_clarity_score,
                sentiment_distribution=sentiment_distribution,
                completeness_score=completeness_score,
                reliability_score=reliability_score
            )
            
            self._save_evaluation_report(metrics)
                      
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise

    def _save_evaluation_report(self, metrics: EvaluationMetrics):
        """
        Save evaluation results to a formatted text file.
        
        Args:
            metrics (EvaluationMetrics): Evaluation results to save
        """
        try:
            report_path = self.output_dir / "evaluation_report.txt"
            
            with open(report_path, "w") as f:
                # Header
                f.write("STAKEHOLDER ANALYSIS EVALUATION REPORT\n")
                f.write("=" * 40 + "\n\n")
                
                # Coverage metrics section
                f.write("COVERAGE METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Potential Stakeholders: {metrics.total_stakeholders}\n")
                f.write(f"Identified Stakeholders: {metrics.identified_count}\n")
                f.write(f"Coverage Rate: {metrics.coverage_rate:.2%}\n\n")
                
                # Relevance metrics section
                f.write("RELEVANCE METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Relevance Score: {metrics.avg_relevance_score:.3f}\n")
                f.write(f"Minimum Relevance Score: {metrics.min_relevance_score:.3f}\n")
                f.write(f"Maximum Relevance Score: {metrics.max_relevance_score:.3f}\n")
                f.write(f"Average Content Chunks per Stakeholder: {metrics.avg_chunks_per_stakeholder:.1f}\n\n")
                
                # Quality metrics section
                f.write("QUALITY METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Content Quality Score: {metrics.content_quality_score:.3f}\n")
                f.write(f"Justification Quality Score: {metrics.justification_quality_score:.3f}\n")
                f.write(f"Impression Clarity Score: {metrics.impression_clarity_score:.3f}\n\n")
                
                # Sentiment analysis section
                f.write("SENTIMENT DISTRIBUTION\n")
                f.write("-" * 20 + "\n")
                for sentiment, count in metrics.sentiment_distribution.items():
                    f.write(f"{sentiment}: {count}\n")
                f.write("\n")
                
                # Overall scores section
                f.write("OVERALL SCORES\n")
                f.write("-" * 20 + "\n")
                f.write(f"Completeness Score: {metrics.completeness_score:.3f}\n")
                f.write(f"Reliability Score: {metrics.reliability_score:.3f}\n\n")
                
                # Save detailed recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                if metrics.coverage_rate < 0.5:
                    f.write("- Consider broadening the content analysis to identify more stakeholders\n")
                if metrics.avg_relevance_score < 0.6:
                    f.write("- Review content matching criteria to improve relevance scores\n")
                if metrics.content_quality_score < 0.6:
                    f.write("- Enhance content collection to gather more relevant information\n")
                if metrics.justification_quality_score < 0.6:
                    f.write("- Improve justification clarity and specificity\n")
                if metrics.impression_clarity_score < 0.6:
                    f.write("- Provide more detailed and specific stakeholder impressions\n")
                
                # Metadata
                f.write("\nEVALUATION METADATA\n")
                f.write("-" * 20 + "\n")
                f.write(f"Evaluation Date: {metrics.evaluation_timestamp}\n")
                
            logger.info(f"Evaluation report saved to {report_path}")
            
            # Save metrics as JSON for programmatic access
            json_path = self.output_dir / "evaluation_metrics.json"
            with open(json_path, "w") as f:
                json.dump({
                    "coverage_metrics": {
                        "total_stakeholders": metrics.total_stakeholders,
                        "identified_count": metrics.identified_count,
                        "coverage_rate": metrics.coverage_rate
                    },
                    "relevance_metrics": {
                        "avg_score": metrics.avg_relevance_score,
                        "min_score": metrics.min_relevance_score,
                        "max_score": metrics.max_relevance_score,
                        "avg_chunks": metrics.avg_chunks_per_stakeholder
                    },
                    "quality_metrics": {
                        "content_quality": metrics.content_quality_score,
                        "justification_quality": metrics.justification_quality_score,
                        "impression_clarity": metrics.impression_clarity_score
                    },
                    "sentiment_distribution": metrics.sentiment_distribution,
                    "overall_scores": {
                        "completeness": metrics.completeness_score,
                        "reliability": metrics.reliability_score
                    },
                    "metadata": {
                        "timestamp": metrics.evaluation_timestamp
                    }
                }, f, indent=2)
            
            logger.info(f"Evaluation metrics saved to {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {str(e)}")
            raise
    
