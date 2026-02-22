"""Reset all ESG platform data and load realistic sample data.

Usage (from project root):

    python scripts/reset_and_seed.py

This will:
- Delete the SQLite DB (`esg_platform.db`)
- Delete evidence files under `evidence_store/`
- Delete the Chroma DB under `chroma_db/`
- Re-run the sample data seeding (framework, entity, IROs, evidence, metric)
"""

import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

DB_PATH = PROJECT_ROOT / "esg_platform.db"


def remove_path(p: Path) -> None:
    if not p.exists():
        return
    if p.is_file():
        p.unlink(missing_ok=True)
    else:
        shutil.rmtree(p, ignore_errors=True)


def main():
    print("Resetting ESG platform data...")

    # 1. Delete DB
    if DB_PATH.exists():
        DB_PATH.unlink(missing_ok=True)
        print(f"  Deleted DB: {DB_PATH.name}")
    else:
        print("  DB not found; nothing to delete.")

    # 2. Delete evidence store and Chroma DB
    remove_path(config.EVIDENCE_STORE_DIR)
    print(f"  Deleted evidence store: {config.EVIDENCE_STORE_DIR}")

    remove_path(config.PERSIST_DIRECTORY)
    print(f"  Deleted vector store (Chroma): {config.PERSIST_DIRECTORY}")

    # 3. Re-seed sample data
    from scripts import seed_sample_data  # type: ignore

    print("\nSeeding fresh sample data...")
    seed_sample_data.main()

    print("\nDone. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()

