"""AI-based and static suggestions for fields where users may not know what to enter."""

from typing import List, Optional

# ---- Static suggestions (no API) ----

FRAMEWORK_SUGGESTIONS = [
    ("CSRD", "EU Corporate Sustainability Reporting Directive"),
    ("GRI", "Global Reporting Initiative Standards"),
    ("TCFD", "Task Force on Climate-related Financial Disclosures"),
    ("SASB", "Sustainability Accounting Standards Board"),
    ("CDP", "CDP (Carbon Disclosure Project)"),
    ("ISSB", "IFRS Sustainability Disclosure Standards"),
]

# Common material topics by framework / general
TOPIC_SUGGESTIONS = [
    "Climate Change",
    "Energy",
    "Water",
    "Biodiversity",
    "Circular Economy",
    "Pollution",
    "Workforce",
    "Human Rights",
    "Governance",
]

SUB_TOPIC_SUGGESTIONS = [
    "GHG Emissions",
    "Scope 1",
    "Scope 2",
    "Scope 3",
    "Energy Consumption",
    "Renewable Energy",
    "Water Withdrawal",
    "Waste",
    "Diversity & Inclusion",
    "Health & Safety",
]

STAKEHOLDER_GROUPS_SUGGESTIONS = [
    "Employees",
    "Investors",
    "Customers",
    "Communities",
    "Regulators",
    "Suppliers",
    "NGOs",
]

# Example questions for Document Q&A when no docs yet / generic
DEFAULT_QA_SUGGESTIONS = [
    "What emission factors are available for common materials?",
    "Summarize Scope 1 and Scope 2 emissions from the documents.",
    "Which activities contribute most to Scope 3 emissions?",
    "What are the key ESG metrics reported?",
]


def get_evidence_tag_suggestions(
    existing_tags: List[str],
    iro_topics: Optional[List[str]] = None,
    year: Optional[str] = None,
) -> List[str]:
    """
    Build suggested evidence tags from existing tags and IRO topics.
    No LLM required; user can click to fill.
    """
    import datetime
    yr = year or str(datetime.datetime.now().year)
    suggestions = list(existing_tags)  # reuse existing
    if iro_topics:
        for topic in iro_topics[:5]:  # limit
            clean = topic.strip().replace(" ", "_")
            suggestions.append(f"CSRD_{yr}_{clean}")
            suggestions.append(f"GRI_{yr}_{clean}")
    if not suggestions:
        suggestions = [
            f"CSRD_{yr}_ClimateChange_Scope1",
            f"CSRD_{yr}_Energy_Consumption",
            f"GRI_{yr}_Emissions",
        ]
    return list(dict.fromkeys(suggestions))  # dedupe, preserve order


def suggest_metric_description_llm(
    evidence_tag: str,
    column_names: List[str],
    sample_preview: str,
    get_llm_func,
    provider_config: dict,
) -> Optional[str]:
    """
    Use LLM to suggest a metric description from tag and sample structure.
    Returns None if LLM unavailable or call fails.
    """
    try:
        llm = get_llm_func(provider_config, temperature=0.2, streaming=False)
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = (
            "You are an ESG reporting assistant. Given an evidence tag and the columns/structure "
            "of an uploaded file, suggest one short metric description that could be computed from this data. "
            "Examples: 'Total Scope 1 GHG emissions (tCO2e)', 'Total electricity consumption (kWh)', "
            "'Total waste generated (tonnes)'. Reply with only the suggested metric description, no explanation."
        )
        text = (
            f"Evidence tag: {evidence_tag}\n"
            f"Columns: {column_names}\n"
            f"Sample (first 500 chars): {sample_preview[:500]}"
        )
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
        content = getattr(response, "content", None) or str(response)
        return (content.strip() or None) if content else None
    except Exception:
        return None


def suggest_qa_questions_llm(
    source_files: List[str],
    get_llm_func,
    provider_config: dict,
    max_questions: int = 5,
) -> List[str]:
    """
    Use LLM to suggest relevant questions based on indexed file names.
    Returns list of suggested questions (or empty list on failure).
    """
    if not source_files:
        return DEFAULT_QA_SUGGESTIONS
    try:
        llm = get_llm_func(provider_config, temperature=0.3, streaming=False)
        from langchain_core.messages import HumanMessage, SystemMessage

        files_preview = ", ".join(source_files[:15])
        prompt = (
            "You are an ESG audit assistant. Based only on these indexed document/file names, "
            f"suggest {max_questions} short, specific questions a user could ask to get useful answers from the documents. "
            "One question per line, no numbering. Focus on emissions, factors, metrics, and evidence."
        )
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"File names: {files_preview}"),
        ])
        content = getattr(response, "content", None) or str(response)
        if not content:
            return DEFAULT_QA_SUGGESTIONS
        lines = [q.strip() for q in content.split("\n") if q.strip()][:max_questions]
        return lines if lines else DEFAULT_QA_SUGGESTIONS
    except Exception:
        return DEFAULT_QA_SUGGESTIONS
