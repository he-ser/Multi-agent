from __future__ import annotations

from pathlib import Path

from customer_service.config import get_settings
from customer_service.graph.workflow import build_customer_service_app


def export_workflow_diagram(output_path: str | Path | None = None) -> Path:
    settings = get_settings()
    target = Path(output_path) if output_path else settings.workflow_diagram_path
    target.parent.mkdir(parents=True, exist_ok=True)

    app = build_customer_service_app()
    graph = app.get_graph()
    if not hasattr(graph, "draw_mermaid"):
        raise RuntimeError("Current langgraph version does not support graph.draw_mermaid().")

    mermaid = graph.draw_mermaid()
    target.write_text(mermaid, encoding="utf-8")
    return target


def main() -> None:
    target = export_workflow_diagram()
    print(f"Workflow diagram exported to {target}")


if __name__ == "__main__":
    main()
