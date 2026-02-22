# ESG Audit Platform — Demo Script for Investors & Clients

**Duration:** 10–15 minutes  
**Audience:** Investors, clients, partners  
**Goal:** Show a single, audit-ready ESG workflow from materiality to reported metrics with full traceability.

---

## Before the Demo

1. **Reset and load sample data (clears everything first):**
   ```bash
   cd C:\Users\anura\Desktop\esg_audit_rag-emission_rag
   python scripts\reset_and_seed.py
   ```

2. **Start the app:**
   ```bash
   streamlit run app.py
   ```

3. **Optional:** Open the app in a clean browser window; zoom so the workflow is visible.

---

## Opening (1 min)

*"We’re going to walk through our **ESG Audit & Assurance Platform** — one place to define what you report, attach evidence, compute metrics, and produce an **audit-ready trail** that investors and auditors expect."*

*"The sample is set up for **GreenTech Industries**, reporting year **2025**, under **CSRD**. You’ll see material topics, evidence files, and a reported Scope 1 emissions number, all linked so every number can be traced back to source documents and the calculation."*

---

## 1. Home — Audit Readiness (2 min)

**Show:** Home page.

*"The dashboard shows **Audit readiness** — a simple progress score based on five steps: framework, entity, material topics, evidence, and reported metrics. With the sample data loaded, we’re at 100%."*

*"The **Workflow** section is the core: **Materiality & Scope** → **Evidence Vault** → **Agentic Auditor** → **Report & Audit**. We’ll follow that order."*

*"There’s also **Document Q&A**: we can index ESG documents and ask questions with **cited sources** — file name, page or sheet — which supports both internal use and auditor checks."*

**Key message for investors:**  
*"One platform from materiality to reported metrics, with traceability built in."*

---

## 2. Materiality & Scope (2–3 min)

**Navigate:** **1. Materiality & Scope**.

*"Step 1 is **Materiality & Scope**. We choose the **reporting framework** — here CSRD — and create a **reporting entity** and **period**. For the demo that’s **GreenTech Industries**, **2025**, status **In progress**."*

*"Then we run a **Double Materiality Assessment**: we define **material topics** — Impacts, Risks, Opportunities — and map each to a **disclosure standard**, e.g. **ESRS E1** for climate, **ESRS E3** for water. That tells auditors exactly which standard each topic supports."*

**Show:** The list of recorded IROs (e.g. Climate Change / Scope 1, Energy / Energy consumption, Water / Water withdrawal) with disclosure codes.

*"So before we collect a single file, we’ve fixed **what** we report and **where** it will show up in the disclosure framework."*

**Key message for clients:**  
*"No more scattered spreadsheets — framework, entity, and material topics are defined and mapped to ESRS/GRI in one place."*

---

## 3. Evidence Vault (2 min)

**Navigate:** **2. Evidence Vault**.

*"Step 2 is the **Evidence Vault**. Every number we report must be backed by **source documents**. Here we upload files and give them a **tag** — e.g. *CSRD_2025_ClimateChange_Scope1* — and optionally **link** them to a material topic. That link is what creates the audit trail."*

**Show:** Evidence by tag (e.g. scope1_emissions_2025.csv, energy_consumption_2025.csv under the sample tags).

*"In the demo we’ve loaded sample CSVs: **Scope 1 emissions** by site and fuel, **energy consumption** by site, and an **emission factors** reference. In production you’d upload real policies, spreadsheets, and reports. The important part is: **one tag per metric/theme**, and **link to the right material topic** so Report & Audit can show coverage."*

**Key message for investors:**  
*"Evidence is stored and tagged; linking to material topics gives us disclosure-level traceability."*

---

## 4. Agentic Auditor (2–3 min)

**Navigate:** **3. Agentic Auditor**.

*"Step 3 is the **Agentic Auditor**. Instead of hand-writing every metric formula, we **select a tag**, pick a **sample file** from that tag, and describe the metric we want — e.g. *Total Scope 1 GHG emissions in tCO2e*. The system **generates** a small Python calculator that reads the same structure and computes the metric. We can **preview** it on the sample file, then **run it on all files** for that tag and **save** the result to the report."*

**Show:** Saved calculator for **CSRD_2025_ClimateChange_Scope1** and the fact that the reported value (e.g. 149.58 tCO2e) is saved and linked to the reporting entity.

*"So the **metric** is defined by **code**; that code is versioned and visible. Auditors can see exactly how the number was derived."*

**Key message for clients:**  
*"We reduce manual formula work and keep a single, auditable definition per metric."*

---

## 5. Report & Audit (2–3 min)

**Navigate:** **4. Report & Audit**.

*"Step 4 is **Report & Audit** — the view we’d give an auditor or an investor asking for proof."*

**Show:** **Reporting period** = GreenTech Industries (2025).

*"First, **Disclosure coverage**: for each material topic we see how many **evidence files** are linked and whether there’s a **reported metric**. So at a glance we see what’s covered and what’s missing."*

**Show:** The audit trail expander for *Total Scope 1 GHG emissions (tCO2e)*.

*"Then the **Audit trail**: for each reported metric we show the **value**, **reporting period**, **source evidence files**, and the **calculation code**. That’s the full chain: **value → evidence → code**. No black box."*

**Key message for investors:**  
*"Disclosure coverage plus a clear audit trail — ready for limited or reasonable assurance."*

---

## Closing & Q&A (1–2 min)

*"So in one platform we: define framework and material topics, attach evidence to those topics, generate and run metric calculators, and produce a report view with disclosure coverage and a full audit trail. All of that is **dynamic** — you can add entities, years, frameworks, and more evidence without starting over."*

**Likely questions**

| Question | Suggested answer |
|----------|-------------------|
| *Can we export to Excel/PDF?* | "The current version focuses on the in-app audit trail. Export (e.g. disclosure table, metric list) can be added as a next phase." |
| *How do you handle multiple entities or consolidation?* | "You create one reporting entity per legal entity or scope. Consolidation and group reporting can be built on top of the same data model." |
| *Is this aligned with CSRD/ESRS?* | "Yes. We use ESRS disclosure codes (E1, E3, etc.) and a double-materiality structure. The platform doesn’t replace legal or assurance advice but supports the data and evidence side." |
| *Where is data stored?* | "Evidence files and database are stored on your infrastructure. No ESG data is sent to third parties except when you use an optional cloud LLM for Document Q&A or calculator generation." |

---

## Sample Data Summary (for reference)

| Item | Example |
|------|--------|
| **Framework** | CSRD |
| **Entity** | GreenTech Industries, 2025, status In progress |
| **IROs** | Climate/Scope 1 (ESRS E1), Energy/Consumption (ESRS E1), Water/Withdrawal (ESRS E3) |
| **Evidence** | scope1_emissions_2025.csv, energy_consumption_2025.csv, emission_factors_ref.csv (under tags CSRD_2025_*) |
| **Reported metric** | Total Scope 1 GHG emissions = 149.58 tCO2e (from sample Scope 1 CSV) |

---

*End of demo script.*
