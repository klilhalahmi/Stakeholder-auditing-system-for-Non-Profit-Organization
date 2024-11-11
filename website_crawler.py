import logging
from bs4 import BeautifulSoup
import requests
import time
from urllib.parse import urljoin, urlparse
from typing import Set, Dict, List, Tuple

from difflib import SequenceMatcher
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from logger_config import CustomLogger

class WebsiteCrawler:
    """
    A comprehensive website crawler that extracts content, creates embeddings, and enables semantic search.
    
    This crawler systematically explores web pages within a specified domain, extracts their content,
    creates text embeddings using OpenAI's embedding model, and stores them in a Chroma vector database
    for semantic similarity search.
    
    Features:
    - Respects domain boundaries and common web crawling conventions
    - Handles rate limiting and timeouts
    - Creates semantic embeddings for content search
    - Provides detailed logging of the crawling process
    - Supports persistent storage of vector embeddings
    
    Args:
        base_url (str): The starting URL for the crawler. Must include protocol (http/https).
        openai_api_key (str): API key for OpenAI's embedding service.
        log_dir (str, optional): Directory for storing log files. Defaults to "logs".
    """
    
    def __init__(self, base_url: str, openai_api_key: str, log_dir: str = "logs"):
        self.base_url = base_url.rstrip('/')
        self.base_domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.website_data = []
        
        # Initialize logger with explicit logging level
        self.logger = CustomLogger(
            log_dir=log_dir,
            log_level=logging.INFO
        ).get_logger("WebsiteCrawler")
        
        self.logger.info(f"Initializing WebsiteCrawler for domain: {self.base_domain}")
        
        # Set up request headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/121.0.0.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        
        try:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            self.logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}", exc_info=True)
            raise
            
        self.vector_store = None

    def is_valid_url(self, url: str) -> bool:
        """
        Validate whether a URL should be crawled based on defined rules.
        
        Rules:
        - URL must be within the same domain as the base URL
        - Excludes common file types (pdf, jpg, png, gif)
        - Excludes special URL patterns (anchors, mailto, tel, javascript)
        
        Args:
            url (str): The URL to validate
            
        Returns:
            bool: True if the URL should be crawled, False otherwise
        """
        try:
            parsed = urlparse(url)
            is_valid = (
                self.base_domain in parsed.netloc and
                not any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.gif']) and
                not any(skip in url.lower() for skip in ['#', 'mailto:', 'tel:', 'javascript:'])
            )
            
            if not is_valid:
                self.logger.debug(f"Skipping invalid URL: {url}")
            
            return is_valid
        except Exception as e:
            self.logger.warning(f"Error validating URL {url}: {str(e)}")
            return False

    def get_page_content(self, url: str) -> Tuple[set, dict]:
        """
        Fetch and parse content from a webpage.
        
        This method:
        1. Retrieves the webpage content with proper encoding
        2. Extracts all valid links for further crawling
        3. Cleans and extracts relevant text content
        4. Handles various exceptions that might occur during the process
        
        Args:
            url (str): The URL to fetch and parse
            
        Returns:
            Tuple[set, dict]: A tuple containing:
                - set: Set of valid URLs found on the page
                - dict: Page data including:
                    - url: The page URL
                    - title: Page title
                    - content: Cleaned text content
        """
        self.logger.info(f"Fetching content from: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.encoding = 'utf-8'
            response.raise_for_status()
            
            self.logger.debug(f"Successfully fetched {url} (Status: {response.status_code})")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract links for crawling
            links = set()
            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                full_url = urljoin(url, href)
                if self.is_valid_url(full_url):
                    links.add(full_url.rstrip('/'))
            
            self.logger.debug(f"Found {len(links)} valid links on {url}")

            # Extract and clean content
            title = soup.title.string if soup.title else ''
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Clean and normalize text content
            text = soup.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = '\n'.join(lines)
            
            self.logger.debug(f"Successfully extracted content from {url} (Title: {title[:50]}...)")
            
            return links, {
                'url': url,
                'title': title,
                'content': content
            }
            
        except requests.Timeout:
            self.logger.error(f"Timeout while fetching {url}")
            return set(), {}
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {str(e)}")
            return set(), {}
        except Exception as e:
            self.logger.error(f"Unexpected error processing {url}: {str(e)}", exc_info=True)
            return set(), {}

    def create_chunks(self) -> List[Document]:
        """
        Create text chunks from crawled content for embedding.
        
        This method:
        1. Splits long text content into smaller, overlapping chunks
        2. Preserves metadata (URL and title) for each chunk
        3. Uses RecursiveCharacterTextSplitter for intelligent text splitting
        
        Returns:
            List[Document]: List of LangChain Document objects containing:
                - page_content: The text chunk
                - metadata: Dictionary with url and title
        """
        self.logger.info("Creating text chunks from crawled content")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        documents = []
        for page in self.website_data:
            try:
                chunks = text_splitter.create_documents(
                    texts=[page['content']],
                    metadatas=[{
                        'url': page['url'],
                        'title': page['title']
                    }]
                )
                documents.extend(chunks)
                self.logger.debug(f"Created {len(chunks)} chunks for {page['url']}")
                
            except Exception as e:
                self.logger.error(f"Error creating chunks for {page['url']}: {str(e)}")
                continue
        
        self.logger.info(f"Created total of {len(documents)} chunks from {len(self.website_data)} pages")
        return documents

    def create_vector_store(self, persist_directory: str = "chroma_db"):
        """
        Create and persist a vector store from the crawled content.
        
        This method:
        1. Creates text chunks from the crawled content
        2. Generates embeddings using OpenAI's embedding model
        3. Stores embeddings in a Chroma vector database
        4. Persists the database to disk for future use
        
        Args:
            persist_directory (str): Directory where the vector store will be saved
        """
        self.logger.info(f"Creating vector store in directory: {persist_directory}")
        
        try:
            documents = self.create_chunks()
            
            if not documents:
                self.logger.error("No documents to create vector store")
                return
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            
            self.vector_store.persist()
            self.logger.info(f"Vector store created successfully with {len(documents)} chunks")
            
        except Exception as e:
            self.logger.error("Failed to create vector store", exc_info=True)
            raise

    def crawl_website(self):
        """
        Main crawling function that orchestrates the entire crawling process.
        
        This method:
        1. Starts with the base URL and discovers new URLs as it crawls
        2. Processes each URL to extract content and find new links
        3. Implements rate limiting to be respectful to the server
        4. Creates a vector store from the crawled content
        5. Provides detailed logging of the crawling progress
        
        Raises:
            KeyboardInterrupt: If the crawl is manually interrupted
            Exception: For any other unexpected errors during crawling
        """
        self.logger.info(f"Starting website crawl for {self.base_url}")
        
        try:
            urls_to_visit = {self.base_url}
            start_time = time.time()
            
            while urls_to_visit:
                current_url = urls_to_visit.pop()
                
                if current_url in self.visited_urls:
                    continue
                
                self.visited_urls.add(current_url)
                self.logger.info(f"Processing URL: {current_url}")
                
                new_links, page_data = self.get_page_content(current_url)
                
                if page_data:
                    self.website_data.append(page_data)
                    self.logger.debug(f"Added content from {current_url} to website data")
                
                urls_to_visit.update(new_links - self.visited_urls)
                
                time.sleep(1)  # Rate limiting
            
            self.create_vector_store()
            
            duration = time.time() - start_time
            self.logger.info(
                f"Crawl completed in {duration:.2f} seconds. "
                f"Processed {len(self.website_data)} pages out of {len(self.visited_urls)} visited URLs."
            )
            
        except KeyboardInterrupt:
            self.logger.warning("Crawl interrupted by user")
            raise
        except Exception as e:
            self.logger.error("Crawl failed with error", exc_info=True)
            raise

    def deduplicate_chunks(self, chunks: List[Dict], max_chunks: int = 5) -> List[Dict]:
            """
            Deduplicate content chunks based on content similarity while maintaining the highest scoring unique chunks.
            
            Args:
                chunks (List[Dict]): List of chunks, each containing 'content' and 'score' keys
                max_chunks (int): Maximum number of unique chunks to return
                
            Returns:
                List[Dict]: Deduplicated chunks, sorted by score
            """
            def normalize_text(text: str) -> str:
                """Normalize text for comparison by removing extra whitespace and converting to lowercase."""
                return re.sub(r'\s+', ' ', text.lower().strip())
            
            def calculate_similarity(text1: str, text2: str) -> float:
                """Calculate similarity ratio between two texts using SequenceMatcher."""
                return SequenceMatcher(None, 
                                    normalize_text(text1),
                                    normalize_text(text2)).ratio()
            
            # Sort chunks by score in descending order
            sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
            unique_chunks = []
            
            # Process each chunk
            for current_chunk in sorted_chunks:
                is_duplicate = False
                
                # Compare with already accepted unique chunks
                for unique_chunk in unique_chunks:
                    similarity = calculate_similarity(current_chunk['content'], 
                                                unique_chunk['content'])
                    
                    if similarity >= 0.9:  # Fixed similarity threshold
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_chunks.append(current_chunk)
                    
                    if len(unique_chunks) >= max_chunks:
                        break
            
            return unique_chunks

    def search_similar_content(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for content similar to the query using semantic similarity.
        
        This method:
        1. Converts the query into an embedding
        2. Finds the most similar content chunks in the vector store using cosine similarity
        3. Uses deduplication to remove similar chunks
        4. Returns the matches with their similarity scores
        
        Args:
            query (str): The search query
            k (int, optional): Number of results to return. Defaults to 3.
            
        Returns:
            List[Dict]: List of dictionaries containing:
                - content: The matched text chunk
                - metadata: Dictionary with url and title
                - score: Similarity score (higher is better)
                
        Raises:
            ValueError: If vector store hasn't been created
            Exception: For any other unexpected errors
        """
        self.logger.info(f"Searching for content similar to: {query}")
        
        try:
            if not self.vector_store:
                error_msg = "Vector store not created. Run crawl_website() first."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Request more chunks than needed to account for potential duplicates
            initial_k = min(k * 2, 20)
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=initial_k)
            
            chunks = [{
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            } for doc, score in results]
            
            # Deduplicate chunks
            unique_chunks = self.deduplicate_chunks(chunks, max_chunks=k)
            
            self.logger.info(f"Found {len(unique_chunks)} unique content chunks after deduplication")
            if unique_chunks:
                self.logger.debug(f"Top match score: {unique_chunks[0]['score']}")
            
            return unique_chunks
            
        except Exception as e:
            self.logger.error(f"Error searching similar content: {str(e)}", exc_info=True)
            raise