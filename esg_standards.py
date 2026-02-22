"""
ESG disclosure standards for dynamic dropdowns (ESRS, GRI, TCFD).
Maps framework names to disclosure codes and descriptions.
"""

# ESRS (EU) – Environmental, Social, Governance
ESRS_DISCLOSURES = [
    ("ESRS 1", "General requirements"),
    ("ESRS 2", "General disclosures"),
    ("ESRS E1", "Climate change"),
    ("ESRS E2", "Pollution"),
    ("ESRS E3", "Water and marine resources"),
    ("ESRS E4", "Biodiversity and ecosystems"),
    ("ESRS E5", "Resource use and circular economy"),
    ("ESRS S1", "Own workforce"),
    ("ESRS S2", "Workers in the value chain"),
    ("ESRS S3", "Affected communities"),
    ("ESRS S4", "Consumers and end-users"),
    ("ESRS G1", "Business conduct"),
]

# GRI – Universal and topic standards (common ones)
GRI_DISCLOSURES = [
    ("GRI 2", "General disclosures"),
    ("GRI 301", "Materials"),
    ("GRI 302", "Energy"),
    ("GRI 303", "Water and effluents"),
    ("GRI 305", "Emissions"),
    ("GRI 306", "Waste"),
    ("GRI 403", "Occupational health and safety"),
    ("GRI 404", "Training and education"),
    ("GRI 405", "Diversity and equal opportunity"),
]

# TCFD – Pillars
TCFD_DISCLOSURES = [
    ("Governance", "Governance of climate-related risks"),
    ("Strategy", "Strategy and climate-related risks"),
    ("Risk management", "Risk management processes"),
    ("Metrics and targets", "Metrics and targets"),
]

# Framework -> list of (code, description)
DISCLOSURES_BY_FRAMEWORK = {
    "CSRD": ESRS_DISCLOSURES,
    "ESRS": ESRS_DISCLOSURES,
    "GRI": GRI_DISCLOSURES,
    "TCFD": TCFD_DISCLOSURES,
    "SASB": [
        ("General", "Industry-specific disclosure topics"),
        ("Environment", "Environmental topics"),
        ("Social", "Social capital topics"),
        ("Governance", "Governance topics"),
    ],
    "CDP": [
        ("Climate", "Climate change questionnaire"),
        ("Water", "Water security questionnaire"),
        ("Forests", "Forests questionnaire"),
    ],
    "ISSB": [
        ("IFRS S1", "General sustainability-related disclosures"),
        ("IFRS S2", "Climate-related disclosures"),
    ],
}

# Audit workflow status (CSRD-style)
AUDIT_STATUSES = [
    ("draft", "Draft"),
    ("in_progress", "In progress"),
    ("under_review", "Under review"),
    ("assured", "Assured"),
]


def get_disclosures_for_framework(framework_name: str):
    """Return list of (code, description) for the framework. Case-insensitive match."""
    if not framework_name:
        return ESRS_DISCLOSURES  # default
    key = framework_name.upper().strip()
    for k, v in DISCLOSURES_BY_FRAMEWORK.items():
        if k in key or key in k:
            return v
    return ESRS_DISCLOSURES
