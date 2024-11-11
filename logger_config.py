import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

class CustomLogger:
    """
    Custom logger configuration that provides both file and console logging capabilities.
    
    This logger implementation features:
    - Rotating file logs with size limits and backup retention
    - Console output for immediate feedback
    - Timestamp-based log file naming
    - Different formatting for file and console outputs
    - Singleton-like behavior to prevent multiple logger instances
    
    Args:
        log_dir (str): Directory where log files will be stored. Defaults to "logs".
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
        
    Example:
        >>> logger = CustomLogger(log_dir="logs", log_level=logging.DEBUG).get_logger("MyApp")
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        # Initialize basic configuration
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self._logger: Optional[logging.Logger] = None
        
        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)
        
        # Create unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"stakeholder_analysis_{timestamp}.log"
        
    def get_logger(self, name: str = "StakeholderAnalysis") -> logging.Logger:
        """
        Get or create a logger instance with the specified configuration.
        
        This method implements a singleton-like pattern to ensure only one logger
        instance exists per name. It configures both file and console handlers
        with appropriate formatters.
        
        Features:
        - Rotating file handler (10MB limit with 5 backups)
        - Console output for development feedback
        - Detailed formatting for file logs
        - Simplified formatting for console logs
        
        Args:
            name (str): Name of the logger instance. Defaults to "StakeholderAnalysis".
            
        Returns:
            logging.Logger: Configured logger instance
            
        Example:
            >>> logger = CustomLogger().get_logger("MyComponent")
            >>> logger.info("Component initialized")
        """
        # Return existing logger if already created
        if self._logger is not None:
            return self._logger
            
        # Create new logger instance
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # Clear any existing handlers to prevent duplicate logging
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Configure formatters with different levels of detail
        file_formatter = logging.Formatter(
            # Detailed format for file logs including file name and line number
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        console_formatter = logging.Formatter(
            # Simplified format for console output
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Configure rotating file handler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB size limit
            backupCount=5           # Keep 5 backup files
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(self.log_level)
        
        # Configure console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(self.log_level)
        
        # Add both handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Store and return logger instance
        self._logger = logger
        return logger

# Create a default logger instance for immediate use
logger = CustomLogger().get_logger()