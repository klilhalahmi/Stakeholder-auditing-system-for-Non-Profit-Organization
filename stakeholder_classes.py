from dataclasses import dataclass
from typing import List

@dataclass(slots=True)
class StakeholderProfile:
    """
    Represents an individual stakeholder type with their specific characteristics.
    This is the data model for each stakeholder, containing all their relevant information.
    """
    name: str
    description: str
    relevancy_parameters: List[str]
    website_interests: List[str]

class StakeholderRegistry:
    """
    Acts as a central registry/factory for all possible stakeholder types.
    This class is responsible for creating and providing access to all stakeholder profiles.
    Think of it as a database or catalog of all possible stakeholders.
    """
    @staticmethod
    def get_all_stakeholders() -> List[StakeholderProfile]:
        return [
            StakeholderProfile(
                name="Staff",
                description="Paid employees who execute programs and operations",
                relevancy_parameters=[
                    "Mentions of job openings",
                    "Employee benefits",
                    "Career development opportunities",
                    "Staff directory or team page"
                ],
                website_interests=[
                    "Career opportunities and job postings",
                    "Employee benefits and culture",
                    "Organization's mission and values",
                    "Professional development opportunities"
                ]
            ),
            
            StakeholderProfile(
                name="Volunteers",
                description="Unpaid supporters who execute programs and operations",
                relevancy_parameters=[
                    "Volunteer opportunities mentioned",
                    "Volunteer recruitment process",
                    "Volunteer impact stories",
                    "Volunteer training programs"
                ],
                website_interests=[
                    "How to get involved",
                    "Volunteer requirements",
                    "Impact measurement",
                    "Training and support provided",
                ]
            ),
            
            StakeholderProfile(
                name="Beneficiaries",
                description="The target population receiving services and programs from the organization",
                relevancy_parameters=[
                    "Description of services offered",
                    "Success stories",
                    "Get support"
                    "Mentorship programs"
                ],
                website_interests=[
                    "Available services and programs",
                    "How to access services",
                    "Success stories and testimonials",
                    "Contact information for support"
                ]
            ),
            
            StakeholderProfile(
                name="Individual Donors",
                description="Private persons who contribute money or assets to support the organization",
                relevancy_parameters=[
                    "Make a Donation",
                    "Impact reporting",
                    "Tax deduction information",
                    "Your gift will help"
                ],
                website_interests=[
                    "Donation methods and options",
                    "Financial transparency",
                    "Impact stories and results",
                    "Recognition programs"
                ]
            ),
            
            StakeholderProfile(
                name="Foundations",
                description="Independent grantmaking institutions that provide structured funding",
                relevancy_parameters=[
                    "Grant application processes",
                    "Program outcomes and impact metrics",
                    "Financial reports",
                    "Partnership opportunities"
                ],
                website_interests=[
                    "Impact measurement and reporting",
                    "Financial transparency",
                    "Organizational capacity",
                    "Strategic planning information"
                ]
            ),
            
            StakeholderProfile(
                name="Corporate Donors",
                description="Businesses that provide financial support through donations or sponsorships",
                relevancy_parameters=[
                    "Corporate partnership programs",
                    "Sponsorship opportunities",
                    "Brand alignment potential",
                    "CSR program alignment",
                    "501(c)(3)"
                ],
                website_interests=[
                    "Partnership opportunities",
                    "Brand visibility options",
                    "Impact metrics",
                    "Recognition programs"
                ]
            ),
            
            StakeholderProfile(
                name="Government Agencies",
                description="Federal, state, and local entities that provide grants and contracts",
                relevancy_parameters=[
                    "Compliance documentation",
                    "Program outcomes",
                    "Public benefit evidence",
                    "Government partnerships"
                ],
                website_interests=[
                    "Compliance information",
                    "Program effectiveness metrics",
                    "Public benefit documentation",
                    "Organizational structure"
                ]
            ),
            
            StakeholderProfile(
                name="Regulators",
                description="Government agencies that enforce laws and regulations to ensure compliance",
                relevancy_parameters=[
                    "Legal compliance information",
                    "Financial transparency",
                    "Board governance",
                    "Annual reports"
                ],
                website_interests=[
                    "Compliance documentation",
                    "Financial reports",
                    "Governance structure",
                    "Program documentation"
                ]
            ),
            
            StakeholderProfile(
                name="Watchdog Organizations",
                description="Independent entities that monitor and evaluate the performance of nonprofits",
                relevancy_parameters=[
                    "Transparency metrics",
                    "Financial reports",
                    "Program effectiveness",
                    "Governance information"
                ],
                website_interests=[
                    "Financial transparency",
                    "Program impact metrics",
                    "Governance documentation",
                    "Operational efficiency"
                ]
            )
        ]
