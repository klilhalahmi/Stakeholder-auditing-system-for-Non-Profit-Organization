import json
from pathlib import Path
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langchain_community.vectorstores import Chroma
from IPython.display import display, Image
from langchain_core.runnables.graph import MermaidDrawMethod
from tqdm import tqdm
from datetime import datetime
from settings import OPENAI_API_KEY, WEBSITE_URL, OUTPUT_DIR, MODEL, RELEVANCE_THRESHOLD, TOTAL_PLAYERS, MAX_RELEVANT
from stakeholder_classes import StakeholderProfile

from website_crawler import WebsiteCrawler
from config import IdentificationState, StakeholderIdentification, StakeholderAudit
from logger_config import logger
from eval import StakeholderAnalysisEvaluator

def check_web_fb(state: IdentificationState) -> IdentificationState:
    """Checks if the URL is from a website or Facebook page."""
    logger.info(f"Checking URL type for: {WEBSITE_URL}")
    
    is_facebook = any(fb_domain in WEBSITE_URL.lower() 
                     for fb_domain in ["facebook.com", "fb.com"])
    
    state.is_facebook = is_facebook
    if is_facebook:
        logger.warning("Facebook URL detected - analysis cannot proceed")
        print("\n===================================")
        print("Facebook Page URL Detected!")
        print("This system currently only supports website analysis.")
        print("Stay tuned for future updates that will include Facebook page analysis.")
        print("===================================\n")
        state.current_step = "facebook"
    else:
        logger.info("Website URL detected - proceeding with analysis")
        state.current_step = "website"
    
    return state

def initialize_website_crawler(state: IdentificationState) -> IdentificationState:
    """Initializes the website crawler and performs initial crawl."""
    crawler = WebsiteCrawler(
        base_url=WEBSITE_URL,
        openai_api_key=OPENAI_API_KEY
    )

    
    # Create a new vector store - if this is a new analysis, uncomment this line
    crawler.crawl_website()

    # # Load existing vector store - if this is a new analysis, comment this line
    # crawler.vector_store = Chroma(
    #     persist_directory="chroma_db",
    #     embedding_function=crawler.embeddings
    # )

    state.crawler = crawler
    state.current_step = "crawl_complete"
    return state

def analyze_stakeholder_relevance(content_chunks: List[Dict], stakeholder: StakeholderProfile) -> tuple[bool, str]:
    """Analyzes if a stakeholder is relevant based on content chunks."""
    logger.info(f"Analyzing relevance for stakeholder: {stakeholder.name}")
    
    try:
        llm = ChatOpenAI(model=MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
        
        # Construct the full prompt
        system_content = f"""You are a website content analyzer focusing on identifying potential stakeholder connections. Your goal is to find any possible relevance between the content and the stakeholder, even minor.

        STAKEHOLDER PROFILE:
        Name: {stakeholder.name}
        Description: {stakeholder.description}
                      
        Consider this stakeholder potentially relevant if you find ANY of these:
        - Direct mentions of the stakeholder type
        - Related activities or programs
        - Indirect references or implications
        - Future plans or potential involvement
        - Similar or related stakeholder groups
        - General organizational focus that aligns with stakeholder interests
        - Content that might interest this stakeholder
        
        The stakeholder typical interests include:
        {chr(10).join(f"- {param}" for param in stakeholder.relevancy_parameters)}
        
        Approach the analysis with these guidelines:
        1. Start by assuming the stakeholder might be relevant
        2. Look for ANY positive indicators, no matter how small
        3. Consider implicit connections and potential relationships
        4. Think about how the content might indirectly relate to the stakeholder

        YOUR RESPONSE MUST BE A VALID JSON OBJECT with this exact format:
        {{"is_relevant": boolean, "justification": "explanation string"}}"""
        
        # Combine content chunks
        combined_content = "\n".join([chunk['content'] for chunk in content_chunks])
        
        human_content = f"""Analyze this content generously, looking for any possible connection to the stakeholder: 

        {combined_content}"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ])
        
        # Get the formatted messages for logging
        formatted_messages = prompt.format_messages(
            stakeholder_name=stakeholder.name,
            stakeholder_description=stakeholder.description,
            relevancy_parameters="\n".join(f"- {param}" for param in stakeholder.relevancy_parameters),
            content=combined_content
        )
        
        response = llm.invoke(formatted_messages)
        
        # Clean the response content of any control characters
        cleaned_content = ''.join(char for char in response.content if ord(char) >= 32 or char in '\n\r\t')
        
        try:
            # Try to parse the cleaned JSON response
            parsed_response = json.loads(cleaned_content)
            return parsed_response["is_relevant"], parsed_response["justification"]
        except json.JSONDecodeError as je:
            # If JSON parsing fails, log the cleaned content and error
            logger.error(f"JSON parsing error for {stakeholder.name}. Cleaned content: {cleaned_content}")
            logger.error(f"JSON error details: {str(je)}")
            
            # Attempt to extract meaningful information from non-JSON response
            # Look for relevance indicators in the raw response
            is_relevant = any(indicator in cleaned_content.lower() 
                            for indicator in ['relevant', 'connection found', 'positive match'])
            
            # Create a simplified response
            return is_relevant, f"Analysis completed but response parsing failed. Raw response available in logs."
            
    except Exception as e:
        logger.error(f"Error analyzing stakeholder {stakeholder.name}: {str(e)}", exc_info=True)
        return False, f"Error during analysis: {str(e)}"

def identify_stakeholders(state: IdentificationState) -> IdentificationState:
    """Identifies relevant stakeholders by analyzing content chunks for each potential stakeholder."""
    identified_stakeholders = []
    stakeholder_data = {}
    justifications = {}
    
    print("\nAnalyzing stakeholders...")
    
    for stakeholder in tqdm(TOTAL_PLAYERS, desc="Analyzing stakeholders"):
        print(f"\nProcessing: {stakeholder.name}")
        
        search_queries = f"{stakeholder.name.lower()}, {stakeholder.description.lower()}"
        all_content_chunks = state.crawler.search_similar_content(search_queries, k=5)
        relevant_chunks = [chunk for chunk in all_content_chunks if chunk['score'] > RELEVANCE_THRESHOLD]
        
        if relevant_chunks:
            is_relevant, justification = analyze_stakeholder_relevance(relevant_chunks, stakeholder)
            
            if is_relevant:
                identified_stakeholders.append(stakeholder)
                stakeholder_data[stakeholder.name] = relevant_chunks
                justifications[stakeholder.name] = justification
                print(f"✓ Relevant - {justification}")
                print("Relevancy parameters found:")
                for param in stakeholder.relevancy_parameters:
                    if any(param.lower() in chunk['content'].lower() for chunk in relevant_chunks):
                        print(f"  - {param}")
            else:
                print(f"✗ Not relevant - {justification}")
        else:
            print(f"✗ No relevant content found for {stakeholder.name}")
    
    state.identified_stakeholders = StakeholderIdentification(
        identified_stakeholders=identified_stakeholders,
        stakeholder_data=stakeholder_data,
        justification=justifications
    )
    
    # Set appropriate state transition
    if not identified_stakeholders:
        print("\n===================================")
        print("No stakeholders were identified!")
        print("Attempting to include core stakeholders as fallback...")
        print("===================================\n")
        state.current_step = "no_stakeholders"
    else:
        print(f"\nSuccessfully identified {len(identified_stakeholders)} stakeholders:")
        for stakeholder in identified_stakeholders:
            print(f"- {stakeholder.name}")
        state.current_step = "identification_complete"
    
    return state

def conduct_stakeholder_audit(state: IdentificationState) -> IdentificationState:
    """Conducts a stakeholder audit by evaluating content sections for each identified stakeholder."""
    llm = ChatOpenAI(model=MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
    all_impressions = {}
    
    logger.info("Starting stakeholder audit process")
    
    for stakeholder in state.identified_stakeholders.identified_stakeholders:
        logger.info(f"Analyzing stakeholder: {stakeholder.name}")
        print(f"\nAnalyzing content for: {stakeholder.name}")
        
        # Format content sections with scores
        stakeholder_chunks = state.identified_stakeholders.stakeholder_data[stakeholder.name]
        sorted_chunks = sorted(stakeholder_chunks, key=lambda x: x['score'], reverse=True)
        content_sections = [
            f"Content Section {i+1} (Relevance Score: {chunk['score']:.2f})\n"
            f"----------------------------------------\n"
            f"{chunk['content']}"
            for i, chunk in enumerate(sorted_chunks)
        ]
        combined_content = "\n\n".join(content_sections)

        # Construct system message with stakeholder context
        system_message = f"""You are a website content analyzer evaluating how well an organization's website serves different stakeholder needs.

        STAKEHOLDER PROFILE:
        Name: {stakeholder.name}
        Description: {stakeholder.description}
        
        KEY INTERESTS:
        {chr(10).join(f"- {interest}" for interest in stakeholder.website_interests)}
        
        TYPICAL NEEDS:
        {chr(10).join(f"- {param}" for param in stakeholder.relevancy_parameters)}

        Evaluate how well the content meets this stakeholder's needs. Consider:
        - Content relevance and completeness
        - Call-to-action effectiveness
        - Value proposition clarity
        - Engagement opportunities
        - Information accessibility
        - Trust indicators"""

        # Construct human message with evaluation structure
        human_message = f"""Analyze this content and provide:

        1. CONTENT EVALUATION
        - Relevance to stakeholder needs:
        - Information completeness:
        - Content clarity and accessibility:
        - Value proposition:

        2. ENGAGEMENT ASSESSMENT
        - Call-to-action effectiveness:
        - Engagement opportunities:
        - Communication channels:
        - Trust indicators:

        3. GAPS AND IMPROVEMENTS
        - Missing critical information:
        - Navigation/accessibility issues:
        - Engagement barriers:
        - Suggested improvements:

        4. OVERALL RATING (1-5 with justification)

        Analyze ONLY the content provided between the triple dashes below. Do not make assumptions or include external information. Base your response exclusively on what is explicitly stated in this content:
        ---
        {combined_content}
        ---
        """

        try:
            result = llm.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ])
            
            all_impressions[stakeholder.name] = result.content
            
            print(f"✓ Analysis completed for {stakeholder.name}")
            print(f"  - Content Sections Analyzed: {len(sorted_chunks)}")
                
        except Exception as e:
            error_msg = f"Error analyzing {stakeholder.name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"! {error_msg}")
            all_impressions[stakeholder.name] = {"error": str(e)}
    
    state.stakeholder_audit = StakeholderAudit(stakeholder_impressions=all_impressions)
    state.current_step = "audit_complete"
    return state

def save_results_node(state: IdentificationState) -> IdentificationState:
    """Saves the final analysis results to files."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "final_audit.txt", "w", encoding="utf-8") as f:
        f.write("STAKEHOLDER AUDIT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for stakeholder in state.identified_stakeholders.identified_stakeholders:
            f.write(f"Audit Results for {stakeholder.name}:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{state.stakeholder_audit.stakeholder_impressions[stakeholder.name]}\n")
            f.write("\n" + "-" * 50 + "\n\n")
    
    with open(output_dir / "stakeholder_analysis_info.txt", "w", encoding="utf-8") as f:
        f.write("STAKEHOLDER ANALYSIS INFORMATION\n")
        f.write("=" * 50 + "\n\n")
        
        for stakeholder in TOTAL_PLAYERS:
            f.write(f"Stakeholder: {stakeholder.name}\n")
            
            search_queries = f"{stakeholder.name.lower()}, {stakeholder.description.lower()}"
            content_chunks = state.crawler.search_similar_content(search_queries, k=MAX_RELEVANT)
            relevant_chunks = [chunk for chunk in content_chunks if chunk['score'] > RELEVANCE_THRESHOLD]
            
            f.write("Retrieved Chunk Scores:\n")
            for i, chunk in enumerate(content_chunks, 1):
                f.write(f"  Chunk {i}: {chunk['score']:.3f}\n")
            
            f.write(f"Total chunks retrieved: {len(content_chunks)}\n")
            f.write(f"Chunks passing threshold: {len(relevant_chunks)}\n")
            
            is_relevant = any(s.name == stakeholder.name 
                            for s in state.identified_stakeholders.identified_stakeholders)
            f.write(f"Relevant for audit: {'Yes' if is_relevant else 'No'}\n")
            f.write("\n" + "-" * 50 + "\n\n")
    
    try:
        evaluator = StakeholderAnalysisEvaluator(output_dir)
        evaluator.evaluate_results(state)
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error("Error during evaluation", exc_info=True)
        print("Evaluation could not be completed, but analysis results were saved.")
    
    state.current_step = "complete"
    return state

def run_identification(website_url: str):
    """Main function to run the stakeholder identification process."""
    logger.info("Starting stakeholder identification process")
    logger.info(f"Analyzing website: {website_url}")
    
    try:
        workflow = StateGraph(IdentificationState)
        
        # Add nodes
        workflow.add_node("check_web_fb", check_web_fb)
        workflow.add_node("initialize_website_crawler", initialize_website_crawler)
        workflow.add_node("identify_stakeholders", identify_stakeholders)
        workflow.add_node("conduct_stakeholder_audit", conduct_stakeholder_audit)
        workflow.add_node("save_analysis_results", save_results_node)
        
        workflow.set_entry_point("check_web_fb")
    
        # Route after URL check
        def route_based_on_url(state: IdentificationState) -> str:
            return state.current_step
        
        workflow.add_conditional_edges(
            "check_web_fb",
            route_based_on_url,
            {
                "website": "initialize_website_crawler",
                "facebook": END
            }
        )
        
        # Route after crawler initialization
        def route_after_crawler(state: IdentificationState) -> str:
            return state.current_step
        
        workflow.add_conditional_edges(
            "initialize_website_crawler",
            route_after_crawler,
            {
                "crawl_complete": "identify_stakeholders"
            }
        )
        
        # Route after identification
        def route_after_identification(state: IdentificationState) -> str:
            return state.current_step
        
        workflow.add_conditional_edges(
            "identify_stakeholders",
            route_after_identification,
            {
                "identification_complete": "conduct_stakeholder_audit",
                "no_stakeholders": END
            }
        )
        
        # Route after audit
        def route_after_audit(state: IdentificationState) -> str:
            return state.current_step
        
        workflow.add_conditional_edges(
            "conduct_stakeholder_audit",
            route_after_audit,
            {
                "audit_complete": "save_analysis_results"
            }
        )
        
        # Final route
        def route_final(state: IdentificationState) -> str:
            return state.current_step
        
        workflow.add_conditional_edges(
            "save_analysis_results",
            route_final,
            {
                "complete": END
            }
        )

        app = workflow.compile()

        try:
            if app.get_graph():
                # Create output directory if it doesn't exist
                output_dir = Path(OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)

                graph_png = app.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API
                )
                
                graph_path = Path(OUTPUT_DIR) / "workflow_graph.png"

                # Save the graph
                graph_path.write_bytes(graph_png)
                
                display(Image(graph_png))
                logger.info("Workflow graph generated successfully")
                
        except Exception as e:
            logger.warning(f"Could not generate/display graph visualization: {e}")
            print("However, workflow will still execute normally.")
        
        initial_state = IdentificationState()
        logger.info("Starting workflow execution")
        app.invoke(initial_state)
        logger.info("Workflow execution completed successfully")
        
    except Exception as e:
        logger.error("Error during workflow execution", exc_info=True)
        raise

if __name__ == "__main__":
    run_identification(WEBSITE_URL)
