"""Agentic Auditor: Selector Agent, Reasoning Agent, and Code Executor."""

import re
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Any, Callable

import db
from evidence_store import get_file_paths_for_tag
from logger import get_logger

log = get_logger()


def extract_python_code(response: str) -> str:
    """Extract Python code from LLM response (markdown code block or raw)."""
    # Try ```python ... ``` first
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ``` ... ```
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def get_emission_factor_from_chroma(material_name: str, factors_vectorstore, k: int = 3) -> Optional[float]:
    """
    Fetch emission factor for a material from the Chroma factors DB (esg_factors collection).
    Queries by material name, then parses the first numeric factor from retrieved chunks
    (e.g. kg CO2e per kg, or tCO2e). Returns None if not found.
    """
    if factors_vectorstore is None:
        return None
    try:
        query = f"emission factor for {material_name} kg CO2e per kg scope 3"
        docs = factors_vectorstore.similarity_search(query, k=k)
        log.info("retrieval_emission_factor | query=%s | k=%s | num_docs=%s", query, k, len(docs) if docs else 0)
        if not docs:
            return None
        combined = " ".join(d.page_content for d in docs)
        # Try to find a number that looks like an emission factor (e.g. 0.1 to 100)
        # Pattern: number possibly with decimal, optional "kg CO2e" / "tCO2e" nearby
        patterns = [
            r"(\d+\.?\d*)\s*(?:kg\s*CO2e|tCO2e|CO2e)\s*(?:per\s*kg)?",
            r"(?:emission\s*factor|factor)[:\s]*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*(?:per\s*kg|/?\s*kg)",
            r"\b(\d+\.?\d+)\b",
        ]
        for pat in patterns:
            m = re.search(pat, combined, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if 1e-4 < val < 1e6:  # plausible range for emission factors
                    log.info("retrieval_emission_factor | material=%s | result=%s", material_name, val)
                    return val
        return None
    except Exception as e:
        log.warning("retrieval_emission_factor | material=%s | error=%s", material_name, e)
        return None


def make_get_emission_factor(factors_vectorstore, embedding_choice: str = None) -> Callable[[str], Optional[float]]:
    """
    Return a callable get_emission_factor(material_name) that queries Chroma (esg_factors)
    and returns the emission factor for that material, or None if missing.
    Use this when running generated calculators so they can multiply material weight by factor.
    """
    if factors_vectorstore is None:
        return lambda material: None

    def get_emission_factor(material_name: str) -> Optional[float]:
        if not material_name or not str(material_name).strip():
            return None
        return get_emission_factor_from_chroma(
            str(material_name).strip(), factors_vectorstore, k=3
        )

    return get_emission_factor


def get_sample_content_from_file(file_path: str) -> str:
    """
    Load a single evidence file and return its content as a string for the Reasoning Agent.
    Used when user selects one file as the sample for code generation (no vectorization).
    """
    df = load_file_to_dataframe(file_path)
    if df is None or df.empty:
        log.info("get_sample_content_from_file | path=%s | empty=True", file_path)
        return ""
    # Stringify: column names + first 100 rows (or all if small) so LLM sees structure
    max_rows = 100
    if len(df) <= max_rows:
        out = f"Columns: {list(df.columns)}\n\n{df.to_string()}"
    else:
        out = f"Columns: {list(df.columns)}\n\nFirst {max_rows} rows:\n{df.head(max_rows).to_string()}\n\n(... {len(df) - max_rows} more rows)"
    log.info("get_sample_content_from_file | path=%s | rows=%s cols=%s content_len=%s", file_path, len(df), len(df.columns), len(out))
    return out


def generate_calculator_code(
    sample_content: str,
    evidence_tag: str,
    metric_description: str,
    llm,
) -> str:
    """
    Reasoning Agent: given sample document content and tag, generate a Python function
    calculate_metric(df) that returns the computed metric from a DataFrame.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at writing Python code to compute sustainability metrics from structured data.
Your task is to write a single Python function named `calculate_metric` that:
- Takes one argument: `df` (a pandas DataFrame). The DataFrame will have the same structure as the sample data provided.
- Returns a single value (int, float, or str) representing the computed metric.
- You may use a function `get_emission_factor(material_name)` that is provided at runtime. It queries the emission factors database (Chroma) and returns the emission factor (float, kg CO2e per kg or similar) for that material, or None if not found. Use it when you need to multiply material weight by emission factor (e.g. for Scope 3: if the data has material type and weight but no pre-calculated emissions, compute emissions as weight * get_emission_factor(material). If get_emission_factor returns None for a material, skip that row or use 0 and add a comment.
- Use only pandas, standard library, and get_emission_factor. No other external APIs.
- Include clear comments explaining the logic.
- Handles missing values and edge cases where possible.
The function must not define get_emission_factor; it is injected when the code runs."""),
        ("human", """Evidence tag: {evidence_tag}
Metric to compute: {metric_description}

Sample data / document content from the evidence vault:
---
{sample_content}
---

Write the Python function `calculate_metric(df)` that, when given a DataFrame with the same structure as this sample, returns the calculated metric. Use get_emission_factor(material_name) when you need emission factors from the database (e.g. Scope 3: weight * get_emission_factor(material)). Output only the function code, optionally inside a markdown code block.""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "evidence_tag": evidence_tag,
        "metric_description": metric_description or f"Metric for {evidence_tag}",
        "sample_content": sample_content[:12000] if sample_content else "(No sample content available)",
    })
    text = getattr(response, "content", None) or str(response)
    code = extract_python_code(text)
    log.info("generate_calculator_code | tag=%s metric=%s | code_len=%s", evidence_tag, metric_description[:80] if metric_description else "", len(code))
    return code


def load_file_to_dataframe(file_path: str) -> pd.DataFrame:
    """Load a single evidence file (Excel, CSV, or PDF-as-text) into a DataFrame."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in (".xlsx", ".xls"):
        # First sheet only for simplicity; could extend to all sheets
        return pd.read_excel(file_path, sheet_name=0, engine="openpyxl" if suffix == ".xlsx" else None)
    if suffix == ".pdf":
        # PDF: build a minimal DataFrame with page content for regex-style extraction
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return pd.DataFrame([{"page_content": d.page_content, "source": d.metadata.get("source", file_path)} for d in docs])
        except Exception:
            return pd.DataFrame({"page_content": [""], "source": [file_path]})
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return pd.DataFrame({"text": [text], "source": [file_path]})
    return pd.DataFrame()


def run_calculator_on_code(
    code_text: str,
    df: pd.DataFrame,
    get_emission_factor: Optional[Callable[[str], Optional[float]]] = None,
) -> Any:
    """
    Execute the generated calculator code in a restricted scope and run calculate_metric(df).
    If get_emission_factor is provided, it is injected so the code can fetch emission factors
    from the Chroma DB (e.g. for Scope 3: weight * get_emission_factor(material)).
    Returns the computed value.
    """
    scope = {"pd": pd, "pandas": pd}
    if get_emission_factor is not None:
        scope["get_emission_factor"] = get_emission_factor
    local: dict = {}
    exec(code_text, scope, local)
    if "calculate_metric" not in local:
        raise ValueError("Generated code did not define calculate_metric")
    result = local["calculate_metric"](df)
    log.info("run_calculator_on_code | df_shape=%s | result=%s", getattr(df, "shape", None), result)
    return result


def run_calculator_on_file(
    code_text: str,
    file_path: str,
    get_emission_factor: Optional[Callable[[str], Optional[float]]] = None,
) -> Tuple[Any, pd.DataFrame]:
    """Load file to DataFrame and run the calculator. Returns (value, df)."""
    df = load_file_to_dataframe(file_path)
    if df.empty:
        return None, df
    value = run_calculator_on_code(code_text, df, get_emission_factor=get_emission_factor)
    return value, df


def run_calculator_on_tag(
    code_text: str,
    evidence_tag: str,
    get_emission_factor: Optional[Callable[[str], Optional[float]]] = None,
) -> Tuple[Any, List[Tuple[str, Any]]]:
    """
    Run the calculator on every file with the given tag.
    If get_emission_factor is provided, generated code can fetch emission factors from Chroma
    (e.g. for Scope 3: material weight * get_emission_factor(material)).
    Returns (aggregated_value, list of (file_path, value) per file).
    """
    file_list = get_file_paths_for_tag(evidence_tag)
    if not file_list:
        return None, []
    results = []
    for path, _ in file_list:
        try:
            val, _ = run_calculator_on_file(code_text, path, get_emission_factor=get_emission_factor)
            results.append((path, val))
        except Exception:
            results.append((path, None))
    # Aggregate: sum if all numeric, else list
    values = [r[1] for r in results if r[1] is not None]
    if not values:
        log.info("run_calculator_on_tag | tag=%s | file_count=%s | aggregated=None (no values)", evidence_tag, len(results))
        return None, results
    try:
        aggregated = sum(values)
    except TypeError:
        aggregated = values
    log.info("run_calculator_on_tag | tag=%s | file_count=%s | aggregated=%s | per_file=%s", evidence_tag, len(results), aggregated, [(p, v) for p, v in results])
    return aggregated, results
