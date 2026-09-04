---
name: ai-endpoint-verification
description: How to verify slt-ai-module /api/ai/* endpoints directly against the real stack (auth, thin-data override, material limitation)
metadata:
  type: reference
---

Verifying Flask `/api/ai/*` endpoints against the real running stack (MySQL `slt_fieldops_db` + Flask on :5000):

- **No JWT enforcement.** `app.py` has no auth middleware on the AI endpoints — direct `curl`/`Invoke-RestMethod` works with any (or no) `Authorization: Bearer` header. You do NOT need a real login token to hit them. (Login at `POST :8080/api/auth/login` `{username:superadmin,password:Admin@2024}` still works and returns a real `SUPER_ADMIN` JWT if you want fidelity.)

- **Thin dev data blocks forecast-dependent endpoints.** The real `faults` table holds only ~8 rows / ~15 filled days. `Config.FORECAST_MIN_HISTORY_DAYS` defaults to 180, so `/api/ai/resource-plan`, `/api/ai/predictions`, `/api/ai/dashboard` return `insufficientData:true` by default. To exercise the populated path, restart Flask with a process env var `FORECAST_MIN_HISTORY_DAYS=5` (ephemeral, no file edit) so the ~15 real days pass. `Config` reads env at import; no `.env` file exists so the process env wins.

- **Material suggestion / shortfall live-path cannot be produced from the real DB.** `material_usage` has 0 rows. Materials only attach to a hotspot when `material_usage -> jobs -> faults(lat/lng)` history exists (`ResourcePlanner._compute_material_rates`), so `hotspots[].materials` and `materialShortfalls` are always empty live, regardless of `materials.current_stock`. The last-resort "set stock to 0" trick does NOT help — a material with no usage history never appears in a hotspot. Verify shortfall logic instead by unit-invoking `ResourcePlanner._compute_shortfalls(hotspots, stock_by_material)` directly with venv python (uses no instance state; `ResourcePlanner(None,None,None)` is fine). Real stock values: id1 CAT6 Cable=100 m, id2 "Cable"=0 pcs.

See [[python-venv-interpreter]] — always use `./venv/Scripts/python.exe`, never bare python.
