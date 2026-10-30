# GUI response schemas

A developer tool for the GUI ([gui/](../../gui/README.md)). The GUI never imports `bio_neb`; it reads each Review's
`response_schema` from `observe`. Its tests need those schemas without a Run, so this tool prints them per Review kind
and action from the backend's response classes, mirroring `src/bio_neb/application/reviews.py` `_legal_actions` with
synthetic request values, plus the finalization files' schemas that `observe` does not publish:

```bash
uv run --locked --offline python -m tools.gui_schemas > gui/tests/schemas.json
```

Rerun it on every M merge into the GUI, and whenever a response class or a finalization file shape changes. A GUI test
compares the GUI's finalization file copies with this output. `src/` never imports this package.
