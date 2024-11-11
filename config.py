from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from website_crawler import WebsiteCrawler

@dataclass(slots=True)
class StakeholderIdentification:
    """
    Data model for storing identified stakeholders and related information.
    
    This class uses slots for memory efficiency and defines the structure for
    stakeholder identification output including the stakeholders found, supporting
    data, and justification for each identification.
    
    Attributes:
        identified_stakeholders (List[str]): List of unique stakeholder names/identifiers
        stakeholder_data (Dict[str, List[Dict]]): Supporting data for each stakeholder,
            where the key is the stakeholder name and the value is a list of relevant
            data points found during analysis
        justification (Dict[str, str]): Reasoning for each stakeholder identification,
            where the key is the stakeholder name and the value is the justification text
    
    Example:
        >>> identification = StakeholderIdentification(
        ...     identified_stakeholders=["Employees", "Customers"],
        ...     stakeholder_data={"Employees": [{"role": "Staff"}]},
        ...     justification={"Employees": "Found in company structure"}
        ... )
    """
    identified_stakeholders: List[str]
    stakeholder_data: Dict[str, List[Dict]]
    justification: Dict[str, str]

@dataclass(slots=True)
class StakeholderAudit:
    """
    Data model for storing stakeholder audit results.
    
    This class captures impressions and analysis results for each identified
    stakeholder based on the audit process.
    
    Attributes:
        stakeholder_impressions (Dict[str, str]): Map of stakeholder names to
            detailed impressions/analysis, where the key is the stakeholder name
            and the value is the audit impression text
    
    Example:
        >>> audit = StakeholderAudit(
        ...     stakeholder_impressions={
        ...         "Employees": "High engagement with company initiatives"
        ...     }
        ... )
    """
    stakeholder_impressions: Dict[str, str]

@dataclass(slots=True)
class IdentificationState:
    """
    State management class for the stakeholder identification process.
    
    This class maintains the current state of the stakeholder identification
    workflow, including crawled data, identification results, and audit status.
    It uses slots for improved memory efficiency and performance.
    
    Attributes:
        website_data (Optional[dict]): Extracted website content and metadata
        crawler (Optional[WebsiteCrawler]): Instance of website crawler used
            for data collection
        identified_stakeholders (Optional[StakeholderIdentification]): Results
            from stakeholder identification process
        stakeholder_audit (Optional[StakeholderAudit]): Results from
            stakeholder audit process
        current_step (str): Current step in the identification workflow.
            Defaults to "start"
        is_facebook (bool): Flag indicating if the source is Facebook.
            Defaults to False
    
    Example:
        >>> state = IdentificationState(
        ...     current_step="identification",
        ...     identified_stakeholders=StakeholderIdentification(
        ...         identified_stakeholders=[],
        ...         stakeholder_data={},
        ...         justification={}
        ...     )
        ... )
    """
    website_data: Optional[dict] = None
    crawler: Optional[WebsiteCrawler] = None
    identified_stakeholders: Optional[StakeholderIdentification] = None
    stakeholder_audit: Optional[StakeholderAudit] = None
    current_step: str = "start"
    is_facebook: bool = False

    def __post_init__(self):
        """
        Post-initialization hook for additional setup.
        Currently used as a placeholder for future initialization needs.
        """
        # This replaces the Pydantic config for allowing arbitrary types
        pass