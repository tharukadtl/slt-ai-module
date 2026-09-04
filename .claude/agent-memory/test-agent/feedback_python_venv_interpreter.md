---
name: python-venv-interpreter
description: Always invoke slt-ai-module Python via ./venv/Scripts/python.exe, never bare python/pytest
metadata:
  type: feedback
---

For any Python in `slt-ai-module` (running `app.py`, throwaway verification scripts, pytest), invoke the venv interpreter explicitly: `./venv/Scripts/python.exe`.

**Why:** bare `python`/`pytest` on this machine resolves to system Python 3.14, not the project's 3.11.9 venv, and fails to import the project's dependencies (pandas, prophet, sqlalchemy, etc.).

**How to apply:** prefix every Python invocation in this module with the venv path. Confirmed working: `./venv/Scripts/python.exe app.py` starts Flask on :5000 with DB connected.
