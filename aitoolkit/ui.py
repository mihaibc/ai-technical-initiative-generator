from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np

from .agent import InitiativeAgent
from .connectors.base import ConnectorConfig
from .connectors.http import HTTPConnector
from .connectors.upload import UploadConnector
from .embeddings import embed_texts
from .prompts import DEFAULT_CATEGORIES
from .storage import (
    Document,
    append_to_index,
    ensure_data_dir,
    load_connections,
    save_connections,
)


def _conn_to_row(c: Dict[str, Any]) -> List[str]:
    return [c.get("name", "(unnamed)"), c.get("type", "?"), json.dumps(c.get("params", {}))[:80]]


def build_ui() -> gr.Blocks:
    ensure_data_dir()
    agent = InitiativeAgent()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # AI-Powered Technical Initiative Generator
        Transform a business objective into a pragmatic, categorized initiative plan.
        """)

        with gr.Tab("Generate Initiatives"):
            obj = gr.Textbox(label="Business Objective", placeholder="e.g., Reduce customer churn by 10%", lines=3)
            with gr.Row():
                use_internal = gr.Checkbox(value=True, label="Use Data Sources")
                use_web = gr.Checkbox(value=False, label="Use Web Research")
            with gr.Row():
                cats = gr.CheckboxGroup(
                    choices=DEFAULT_CATEGORIES,
                    value=DEFAULT_CATEGORIES,
                    label="Categories",
                )
                num_per = gr.Slider(1, 5, value=3, step=1, label="Initiatives per Category")
            constraints = gr.Textbox(label="Constraints (optional)", placeholder="comma-separated constraints", lines=2)
            run = gr.Button("Generate", variant="primary")
            with gr.Row():
                out_md = gr.Markdown(label="Plan", elem_id="output-md")
            with gr.Accordion("Details", open=False):
                dbg = gr.Textbox(label="Debug Prompt", lines=6)
                ctx = gr.JSON(label="Generation Context")

            def _on_generate(o, ui, ui_flag, uw_flag, cs, n):
                parsed_constraints = [c.strip() for c in (cs or "").split(",") if c.strip()]
                res = agent.generate(
                    objective=o,
                    categories=ui,
                    use_internal=bool(ui_flag),
                    use_web=bool(uw_flag),
                    constraints=parsed_constraints,
                    num_per_category=int(n),
                )
                return res["output_markdown"], res["prompt"], {
                    "used_internal": res["used_internal"],
                    "used_web": res["used_web"],
                    "context_count": res["context_count"],
                }

            # Note: order of args must match function
            run.click(_on_generate, [obj, cats, use_internal, use_web, constraints, num_per], [out_md, dbg, ctx])

        with gr.Tab("Data Sources"):
            gr.Markdown("Add and manage connections to multiple data sources.")
            with gr.Row():
                table = gr.Dataframe(
                    headers=["Name", "Type", "Params"],
                    value=[_conn_to_row(c) for c in load_connections()],
                    interactive=False,
                    label="Configured Connections",
                    row_count=(len(load_connections()) or 1),
                    col_count=(3),
                )
            with gr.Row():
                with gr.Column():
                    mode = gr.Radio(["Upload", "URL"], value="Upload", label="Connection Type")
                    name = gr.Textbox(label="Name", placeholder="e.g., Support CSV")
                    upload = gr.File(label="Upload (CSV/JSON/TXT)", file_types=[".csv", ".json", ".txt"], type="filepath", visible=True)
                    url = gr.Textbox(label="URL", visible=False)
                    auth = gr.Dropdown(["none", "basic", "bearer"], value="none", label="Auth", visible=False)
                    username = gr.Textbox(label="Username", visible=False)
                    password = gr.Textbox(label="Password", type="password", visible=False)
                    token = gr.Textbox(label="Token", type="password", visible=False)
                    add_btn = gr.Button("Add Source", variant="primary")

                def _toggle_fields(m):
                    if m == "Upload":
                        return (
                            gr.update(visible=True),  # upload
                            gr.update(visible=False), # url
                            gr.update(visible=False), # auth
                            gr.update(visible=False), # username
                            gr.update(visible=False), # password
                            gr.update(visible=False), # token
                        )
                    else:
                        return (
                            gr.update(visible=False), # upload
                            gr.update(visible=True),  # url
                            gr.update(visible=True),  # auth
                            gr.update(visible=True),  # username
                            gr.update(visible=True),  # password
                            gr.update(visible=True),  # token
                        )
                mode.change(_toggle_fields, [mode], [upload, url, auth, username, password, token])

                def _add_source(m, n, f, u, a, un, pw, tk):
                    conns = load_connections()
                    if m == "Upload":
                        if f is None:
                            raise gr.Error("Please upload a file.")
                        # f is a filepath
                        import os, mimetypes
                        filename = os.path.basename(f)
                        with open(f, "rb") as fh:
                            content = fh.read()
                        mime = mimetypes.guess_type(filename)[0] or "text/plain"
                        cfg = ConnectorConfig(
                            type="upload",
                            name=n or filename,
                            params={
                                "filename": filename,
                                "mime": mime,
                                "content": content,
                            },
                        )
                        docs = UploadConnector(cfg).fetch()
                    else:
                        if not u:
                            raise gr.Error("Please provide a URL.")
                        cfg = ConnectorConfig(
                            type="http",
                            name=n or u,
                            params={
                                "url": u,
                                "auth_type": a or "none",
                                "username": un or "",
                                "password": pw or "",
                                "token": tk or "",
                            },
                        )
                        docs = HTTPConnector(cfg).fetch()

                    # Embed and append
                    vectors = embed_texts([d.text for d in docs])
                    append_to_index(docs, vectors)

                    # Save connection (without raw content)
                    safe_params = dict(cfg.params)
                    if safe_params.get("content"):
                        safe_params["content"] = f"<bytes:{len(safe_params['content'])}>"
                    conns.append({"type": cfg.type, "name": cfg.name, "params": safe_params})
                    save_connections(conns)
                    return [
                        _conn_to_row(c) for c in conns
                    ], gr.update(row_count=len(conns))

                add_btn.click(
                    _add_source,
                    [mode, name, upload, url, auth, username, password, token],
                    [table, table],
                )

        with gr.Tab("About"):
            gr.Markdown(
                """
                ### The Engineering Leader's AI Toolkit — Pillar 2
                This Space turns a business objective into a categorized, actionable initiative plan.

                - Diagnostics (Pillar 1): Code Review Quality Analyzer
                - Strategy (Pillar 2): Technical Initiative Generator (this Space)
                - People & Culture (Pillar 3): Coming soon

                Configure data sources and optionally enable web research to ground the plan in reality.
                """
            )

    return demo
