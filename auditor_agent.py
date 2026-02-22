"""Agentic Auditor: materiality-aware, query-planning, retrieval, and code generation."""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Any, Callable, Dict

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


def make_get_emission_factor(factors_vectorstore) -> Callable[[str], Optional[float]]:
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


def get_materiality_context(evidence_tag: str, framework_id: Optional[int] = None) -> str:
    """
    Build a materiality context string for the auditor agent: double materiality (IROs)
    linked to this evidence tag or to the framework. Used to align metric logic with
    disclosure scope (e.g. ESRS E1, Scope 1/2/3).
    """
    iros = db.get_iros_for_evidence_tag(evidence_tag)
    if not iros and framework_id is not None:
        iros = db.list_iros(framework_id=framework_id)
    if not iros:
        return "No materiality (IRO) context found for this tag or framework. Proceed from sample data and metric description only."
    lines = [
        "Double materiality (IROs) relevant to this evidence / metric:",
        "- Impact = inside-out; Financial = outside-in; Both = double materiality.",
    ]
    for iro in iros:
        line = f"- Topic: {iro['topic']} | Sub-topic: {iro['sub_topic']} | Scope: {iro['materiality_scope']}"
        if iro.get("disclosure_code"):
            line += f" | Disclosure: {iro['disclosure_code']}"
        if iro.get("description"):
            line += f" | Notes: {iro['description'][:200]}"
        lines.append(line)
    log.info("get_materiality_context | tag=%s framework_id=%s | iro_count=%s", evidence_tag, framework_id, len(iros))
    return "\n".join(lines)


def plan_emission_factor_queries(
    sample_content: str,
    materiality_context: str,
    llm,
) -> List[Dict[str, Any]]:
    """
    Agent step: from sample data and materiality, decide which materials/activities need
    emission factors and prepare queries for the vector store (material/activity name, unit, scope).
    Returns a list of dicts: [{"material_or_activity": str, "unit": str, "scope": str, "query": str}, ...].
    """
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an ESG auditor agent. Given sample data (columns and sample rows) and double materiality context, identify which materials or activities will need an emission factor from the factors database (e.g. fuels, electricity, materials by weight).
Output a JSON array of objects. Each object must have:
- "material_or_activity": exact name or short phrase as it appears in the data or standard (e.g. "Natural gas", "Diesel (stationary)", "Electricity (grid)", "Steel (recycled)").
- "unit": unit of consumption/quantity in the data (e.g. "GJ", "MWh", "kg").
- "scope": "Scope 1", "Scope 2", or "Scope 3" if evident from materiality or data; otherwise "Unknown".
- "query": a short search query to look up the emission factor in a vector store (e.g. "Natural gas GJ emission factor Scope 1" or "Steel recycled kg CO2e per kg Scope 3").

If the sample data has no columns that suggest fuel type, material type, or activity type, return an empty array [].
Output only the JSON array, no markdown or explanation."""),
        ("human", """Materiality context:
{materiality_context}

Sample data (columns and sample rows):
---
{sample_content}
---

JSON array of emission factor lookup queries:"""),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "materiality_context": materiality_context[:4000],
        "sample_content": (sample_content or "")[:8000],
    })
    text = getattr(response, "content", None) or str(response)
    text = text.strip()
    # Strip markdown code block if present
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        queries = json.loads(text)
        if not isinstance(queries, list):
            queries = []
    except Exception:
        queries = []
    log.info("plan_emission_factor_queries | num_queries=%s", len(queries))
    return queries


def retrieve_emission_factors_for_queries(
    queries: List[Dict[str, Any]],
    factors_vectorstore,
    k: int = 3,
) -> str:
    """
    Run prepared queries against the emission factors vector store and return a single
    text summary (snippets) for the code generator, so it knows which material names
    and units exist and can align get_emission_factor(material) calls.
    """
    if not factors_vectorstore or not queries:
        return ""
    snippets = []
    for q in queries:
        query_str = q.get("query") or f"{q.get('material_or_activity', '')} {q.get('unit', '')} emission factor"
        if not query_str.strip():
            continue
        try:
            docs = factors_vectorstore.similarity_search(query_str.strip(), k=k)
            for d in docs:
                snippets.append(d.page_content)
        except Exception as e:
            log.warning("retrieve_emission_factors_for_queries | query=%s | error=%s", query_str[:80], e)
    if not snippets:
        return ""
    # Dedupe and limit size
    seen = set()
    unique = []
    for s in snippets:
        if s not in seen and len(s) > 10:
            seen.add(s)
            unique.append(s)
    combined = "\n\n---\n\n".join(unique[:15])
    log.info("retrieve_emission_factors_for_queries | queries=%s | snippets=%s", len(queries), len(unique))
    return combined


def get_emission_factor_from_chroma_with_query(query: str, factors_vectorstore, k: int = 3) -> Optional[float]:
    """
    Fetch emission factor using a prepared query string (e.g. "Natural gas GJ Scope 1").
    Parses numeric factor from retrieved chunks. Used when we have a planned query.
    """
    if factors_vectorstore is None or not query or not query.strip():
        return None
    try:
        docs = factors_vectorstore.similarity_search(query.strip(), k=k)
        if not docs:
            return None
        combined = " ".join(d.page_content for d in docs)
        patterns = [
            r"(\d+\.?\d*)\s*(?:kg\s*CO2e|tCO2e|CO2e)\s*(?:per\s*(?:kg|unit|GJ|MWh))?",
            r"(?:emission\s*factor|factor)[:\s]*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*(?:per\s*kg|/?\s*kg|/?\s*GJ|/?\s*MWh)",
            r"\b(\d+\.?\d+)\b",
        ]
        for pat in patterns:
            m = re.search(pat, combined, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if 1e-4 < val < 1e6:
                    return val
        return None
    except Exception as e:
        log.warning("get_emission_factor_from_chroma_with_query | query=%s | error=%s", query[:80], e)
        return None


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
    materiality_context: Optional[str] = None,
    emission_factor_hints: Optional[str] = None,
) -> str:
    """
    Reasoning Agent: given sample content, tag, and optional materiality + emission-factor
    hints, generate a Python function calculate_metric(df) that returns the computed metric.
    """
    from langchain_core.prompts import ChatPromptTemplate

    materiality_block = ""
    if materiality_context:
        materiality_block = f"""Materiality / disclosure context (use to align metric with Scope and disclosure):
---
{materiality_context[:3000]}
---
"""

    hints_block = ""
    if emission_factor_hints:
        hints_block = f"""Retrieved emission factor snippets from the vector store (use these material/activity names when calling get_emission_factor so lookup succeeds):
---
{emission_factor_hints[:4000]}
---
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at writing Python code to compute sustainability metrics from structured data and double materiality (ESG/CSRD) context.
Your task is to write a single Python function named `calculate_metric` that:
- Takes one argument: `df` (a pandas DataFrame). The DataFrame will have the same structure as the sample data provided.
- Returns a single value (int, float, or str) representing the computed metric. Prefer numeric results (e.g. sum of emissions) with a clear unit implied by the metric description.
- You may use a function `get_emission_factor(material_name)` that is provided at runtime. It queries the emission factors database (Chroma) and returns the emission factor (float, kg CO2e per unit) for that material/activity, or None if not found. When the sample has material/activity type and quantity (e.g. GJ, MWh, kg), compute emissions as quantity * get_emission_factor(material_name). Use the exact material/activity names that appear in the retrieved emission factor snippets when possible, so the vector store lookup succeeds.
- Use only pandas, standard library, and get_emission_factor. No other external APIs.
- Include clear comments explaining the logic.
- Check that required columns exist, handle NaN/missing values, and use consistent units (e.g. tCO2e if the metric description says so). Do not assume column names beyond the sample; normalize case or strip whitespace if needed.
The function must not define get_emission_factor; it is injected when the code runs."""),
        ("human", """Evidence tag: {evidence_tag}
Metric to compute: {metric_description}
{materiality_block}
{hints_block}
Sample data / document content from the evidence vault:
---
{sample_content}
---

Write the Python function `calculate_metric(df)` that, when given a DataFrame with the same structure as this sample, returns the calculated metric. Use get_emission_factor(material_name) with material names that match the emission factor database. Output only the function code, optionally inside a markdown code block."""),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "evidence_tag": evidence_tag,
        "metric_description": metric_description or f"Metric for {evidence_tag}",
        "materiality_block": materiality_block,
        "hints_block": hints_block,
        "sample_content": sample_content[:12000] if sample_content else "(No sample content available)",
    })
    text = getattr(response, "content", None) or str(response)
    code = extract_python_code(text)
    log.info("generate_calculator_code | tag=%s metric=%s | code_len=%s", evidence_tag, metric_description[:80] if metric_description else "", len(code))
    return code


def run_auditor_agent(
    evidence_tag: str,
    sample_file_path: str,
    metric_description: str,
    llm,
    framework_id: Optional[int] = None,
    factors_vectorstore=None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Agentic pipeline: understand materiality → load sample → plan emission-factor queries
    → retrieve from vector store → generate calculator code. Returns (generated_code, steps_log).
    """
    steps_log: List[Dict[str, Any]] = []

    # Step 1: Materiality context
    materiality_context = get_materiality_context(evidence_tag, framework_id)
    steps_log.append({"step": "materiality", "summary": "Understood double materiality (IROs) and disclosure scope.", "detail": materiality_context[:500]})

    # Step 2: Sample data
    sample_content = get_sample_content_from_file(sample_file_path)
    if not sample_content:
        steps_log.append({"step": "sample", "summary": "No sample content could be loaded.", "detail": ""})
        return "", steps_log
    steps_log.append({"step": "sample", "summary": "Loaded sample file structure and rows for code generation.", "detail": sample_content[:800]})

    # Step 3: Plan queries for emission factors
    emission_queries = plan_emission_factor_queries(sample_content, materiality_context, llm)
    if emission_queries:
        steps_log.append({
            "step": "query_plan",
            "summary": f"Prepared {len(emission_queries)} emission factor query(ies) for the vector store.",
            "detail": json.dumps(emission_queries, indent=2),
        })
    else:
        steps_log.append({"step": "query_plan", "summary": "No emission factor queries needed for this sample.", "detail": ""})

    # Step 4: Retrieve from vector store
    emission_factor_hints = retrieve_emission_factors_for_queries(emission_queries, factors_vectorstore)
    if emission_factor_hints:
        steps_log.append({"step": "retrieval", "summary": "Retrieved emission factor snippets from vector store.", "detail": emission_factor_hints[:600]})
    else:
        steps_log.append({"step": "retrieval", "summary": "No emission factor DB or no snippets retrieved.", "detail": ""})

    # Step 5: Generate code
    code = generate_calculator_code(
        sample_content,
        evidence_tag,
        metric_description or f"Metric for {evidence_tag}",
        llm,
        materiality_context=materiality_context,
        emission_factor_hints=emission_factor_hints or None,
    )
    steps_log.append({"step": "code_gen", "summary": "Generated calculator code from sample + materiality + emission hints.", "detail": code[:800] if code else ""})

    return code, steps_log


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
    if suffix == ".json":
        # JSON: expect either a list of row dicts or {"rows": [...]} structure
        import json
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return pd.DataFrame()
        if isinstance(data, dict) and "rows" in data:
            data = data.get("rows") or []
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()
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
