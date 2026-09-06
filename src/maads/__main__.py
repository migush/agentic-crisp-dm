"""CLI entry point: `python -m maads ...`

Subcommands:
    data download --case <name>         Download a bundled demonstration dataset.
    data download --competition <slug>  Download any Kaggle competition.
    run --case <name>                   Run the CrewAI-backed CRISP-DM pipeline.
    run --config <path>                 Run from an explicit case config YAML.
    flow plot                           Visualize the CRISP-DM Flow graph.
    dashboard                           Web UI for live trace monitoring.
    artifacts render --run <path>       Regenerate trace markdown/mermaid views.
    artifacts backfill-timing         Backfill duration on existing run artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from maads.artifact_runs import case_root, prepare_run_dir
from maads.artifact_paths import ensure_run_layout
from maads.knowledge_setup import repo_root
from maads.model_capabilities import log_model_capabilities
from maads.model_info import log_selected_model_info
from maads.rag import ensure_embedding_model_available

from maads.config import load_case_config, kickoff_inputs
from maads.data_utils import download_case_data, download_kaggle_competition
from maads.observability import auto_enable, begin_run, configure_crewai_runtime, end_run
from maads.paths import resolve_path
from maads.progress import start_run as start_progress, stop_run as stop_progress
from maads.run_status import bind_run, flush_status
from maads.shutdown import apply_interrupt_to_state, install_sigint_handler, shutdown_requested
from maads.outcome import ml_outcome_deficits, ml_run_succeeded, workflow_complete
from maads.state import CrispDMState


def _persist_run_outcome(
    state: CrispDMState,
    artifact_dir: Path,
    case_dir: Path,
) -> tuple[Path, "RunPaths"]:
    """Write final state and reports; safe to call from ``finally``."""
    from maads.artifact_paths import RunPaths
    from maads.dashboard.store import read_communications
    from maads.observability.collector import get_collector
    from maads.reports import write_run_reports

    state_path = artifact_dir / "final_state.json"
    state_path.write_text(state.model_dump_json(indent=2))
    paths = RunPaths(artifact_dir)
    try:
        trace = get_collector().to_trace_run()
        comms = read_communications(artifact_dir)
        write_run_reports(
            state,
            artifact_dir,
            trace=trace,
            communications=comms,
            case_root=case_dir,
        )
    except Exception as exc:
        print(f"WARNING: report generation failed: {exc}", file=sys.stderr)
    return state_path, paths


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    configure_crewai_runtime()
    auto_enable()

    parser = argparse.ArgumentParser(prog="maads")
    sub = parser.add_subparsers(dest="cmd")

    # ── run ────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run the agentic pipeline on a dataset.")
    g_run = p_run.add_mutually_exclusive_group(required=True)
    g_run.add_argument("--case", help="Shorthand: load configs/<case>.yaml.")
    g_run.add_argument("--config", type=Path,
                       help="Path to a case config YAML.")
    p_run.add_argument("--config-dir", default="configs",
                       help="Directory holding bundled <case>.yaml files. "
                            "Used only when --case is given.")
    p_run.add_argument("--artifact-dir", default="artifacts",
                       help="Root for per-case artefacts. Each run writes to "
                            "artifacts/<case_id>/runs/<run_id>/; prior runs "
                            "are kept under artifacts/<case_id>/archive/.")
    p_run.add_argument("--quiet", "-q", action="store_true",
                       help="Disable live progress output (also MAADS_PROGRESS=0).")
    p_run.add_argument("--model", default=None,
                       help="Override the MODEL env for this run "
                            "(e.g. ollama/gpt-oss:120b-cloud, gpt-4o). "
                            "Recorded in the run manifest.")
    p_run.add_argument("--max-tokens", type=int, default=None,
                       help="Override MAX_TOKENS_PER_RUN for this run (hard token cap).")

    # ── data ───────────────────────────────────────────────────────────
    p_data = sub.add_parser("data", help="Data utilities.")
    sub_data = p_data.add_subparsers(dest="data_cmd")

    p_dl = sub_data.add_parser("download",
                               help="Download a Kaggle competition's data.")
    g_dl = p_dl.add_mutually_exclusive_group(required=True)
    g_dl.add_argument("--case", help="Shorthand for bundled configs.")
    g_dl.add_argument("--competition",
                      help="Any Kaggle competition slug.")
    p_dl.add_argument("--out-dir", default=None,
                      help="Where to put the data (default: data/<name>/).")

    # ── flow ───────────────────────────────────────────────────────────
    p_flow = sub.add_parser("flow", help="CrewAI Flow utilities.")
    sub_flow = p_flow.add_subparsers(dest="flow_cmd")
    p_flow_plot = sub_flow.add_parser("plot", help="Render the CRISP-DM Flow graph.")
    p_flow_plot.add_argument(
        "--output",
        default="crisp_dm_flow",
        help="Output path prefix for the HTML graph (default: crisp_dm_flow).",
    )

    # ── dashboard ──────────────────────────────────────────────────────
    p_dash = sub.add_parser("dashboard", help="Launch the trace monitoring web UI.")
    p_dash.add_argument("--case", default=None,
                        help="Open a specific case (e.g. titanic).")
    p_dash.add_argument("--artifact-dir", default="artifacts",
                        help="Root directory containing per-case artifact folders.")
    p_dash.add_argument("--port", type=int, default=8765,
                        help="HTTP port (default: 8765).")
    p_dash.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1).")
    p_dash.add_argument("--no-open", action="store_true",
                        help="Do not open a browser tab automatically.")
    p_dash.add_argument("--static-dir", type=Path, default=None,
                        help="Serve built frontend from this directory "
                             "(default: dashboard/dist if it exists).")

    # ── artifacts ──────────────────────────────────────────────────────
    p_art = sub.add_parser("artifacts", help="Artifact utilities.")
    sub_art = p_art.add_subparsers(dest="artifacts_cmd")
    p_art_render = sub_art.add_parser(
        "render", help="Regenerate trace markdown/mermaid from trace.json.",
    )
    p_art_render.add_argument(
        "--run", type=Path, required=True,
        help="Run directory (artifacts/<case>/runs/<run_id>).",
    )
    p_art_backfill = sub_art.add_parser(
        "backfill-timing",
        help="Backfill duration_ms and timestamps on existing run artifacts.",
    )
    p_art_backfill.add_argument(
        "--artifact-dir", default="artifacts",
        help="Root directory containing per-case artifact folders.",
    )
    p_art_backfill.add_argument(
        "--run", type=Path, default=None,
        help="Optional single run directory to backfill.",
    )
    p_art_backfill.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing files.",
    )

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0
    if args.cmd == "data" and args.data_cmd is None:
        p_data.print_help()
        return 0

    log_model_capabilities()

    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "flow" and args.flow_cmd == "plot":
        return cmd_flow_plot(args)
    if args.cmd == "flow" and args.flow_cmd is None:
        p_flow.print_help()
        return 0
    if args.cmd == "artifacts" and args.artifacts_cmd == "render":
        return cmd_artifacts_render(args)
    if args.cmd == "artifacts" and args.artifacts_cmd == "backfill-timing":
        return cmd_artifacts_backfill_timing(args)
    if args.cmd == "artifacts" and args.artifacts_cmd is None:
        p_art.print_help()
        return 0
    if args.cmd == "data" and args.data_cmd == "download":
        return cmd_data_download(args)
    if args.cmd == "dashboard":
        return cmd_dashboard(args)
    parser.error("unreachable")
    return 2  # pragma: no cover


def cmd_run(args: argparse.Namespace) -> int:
    """Resolve the case config and run the CRISP-DM Flow pipeline."""
    import os

    os.chdir(repo_root())
    # Set MODEL before the embedding probe and manifest write so both see the
    # UI/CLI-selected model. load_dotenv ran earlier with override=False, so it
    # will not clobber this. MAADS_MODEL_OVERRIDE makes the chosen model
    # authoritative for every agent, beating any per-role MODEL_* in .env.
    model_arg = (getattr(args, "model", None) or "").strip()
    if model_arg:
        os.environ["MODEL"] = model_arg
        os.environ["MAADS_MODEL_OVERRIDE"] = model_arg
        log_selected_model_info(model_arg)
    max_tokens_arg = getattr(args, "max_tokens", None)
    if max_tokens_arg is not None:
        os.environ["MAX_TOKENS_PER_RUN"] = str(max_tokens_arg)
    embed_warn = ensure_embedding_model_available()
    if embed_warn:
        print(f"WARNING: {embed_warn}", file=sys.stderr)

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = resolve_path(config_path)
    else:
        config_path = resolve_path(args.config_dir) / f"{args.case}.yaml"

    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_case_config(config_path)
    state = CrispDMState.from_config(config)

    case_dir = case_root(resolve_path(args.artifact_dir), config.case_id)
    run_id = str(uuid.uuid4())
    artifact_dir = prepare_run_dir(case_dir, run_id)
    ensure_run_layout(artifact_dir, run_id=run_id, case_id=config.case_id,
                      model=os.environ.get("MODEL"))

    # Unconditional per-run assignment: concurrent runs of the same case must not
    # share one CrewAI store, so do not honour an inherited CREWAI_STORAGE_DIR.
    os.environ["CREWAI_STORAGE_DIR"] = str((artifact_dir / "crewai_storage").resolve())

    inputs = kickoff_inputs(config)
    (artifact_dir / "kickoff_inputs.json").write_text(
        json.dumps(inputs, indent=2), encoding="utf-8",
    )

    resolved = artifact_dir.resolve()
    print(f"Case root:   {case_dir.resolve()}")
    print(f"Run ID:      {run_id}")
    print(f"Artifacts:   {resolved}")
    print(f"Archive:     {(case_dir / 'archive').resolve()}")
    print(f"Live status: {resolved / 'status.json'}  (refresh while running)")
    print(f"Live summary: {resolved / 'derived' / 'live_summary.json'}")
    print(f"Trace:       {resolved / 'trace'}/")
    print(f"Exec scripts: {resolved / 'sandbox' / 'exec'}/")

    install_sigint_handler()
    bind_run(artifact_dir, state)
    start_progress(config.case_id, quiet=args.quiet)
    begin_run(config.case_id, artifact_dir, run_id=run_id)
    halt_reason: str | None = None
    interrupted = False
    run_failed = False
    state_path: Path | None = None
    paths = None
    try:
        from maads.flow.crisp_dm_flow import CrispDMFlow

        state = CrispDMFlow(state, artifact_dir).run()
        if not state.halted and not workflow_complete(state):
            from maads.flow.phase_runner import force_halt

            force_halt(state, "flow ended before workflow completion")
        if shutdown_requested() and not state.halted:
            halt_reason = apply_interrupt_to_state(state)
            interrupted = True
        else:
            halt_reason = state.halt_reason
    except KeyboardInterrupt:
        halt_reason = apply_interrupt_to_state(state)
        interrupted = True
        print("\nForced quit.", file=sys.stderr)
    except Exception as exc:
        run_failed = True
        if not state.halted:
            from maads.flow.phase_runner import force_halt

            force_halt(state, f"run failed: {type(exc).__name__}: {exc}")
        halt_reason = state.halt_reason
        print(f"\nRun failed: {exc}", file=sys.stderr)
    finally:
        flush_status()
        end_run(artifact_dir)
        stop_progress(halt_reason, ml_success=ml_run_succeeded(state))
        state_path, paths = _persist_run_outcome(state, artifact_dir, case_dir)

    print(f"Halted: {state.halt_reason}")
    print(f"Workflow: {'complete' if workflow_complete(state) else 'incomplete'}")
    ml_ok = ml_run_succeeded(state)
    deficits = ml_outcome_deficits(state)
    if ml_ok:
        print("ML outcome: success")
    else:
        print(f"ML outcome: failed ({'; '.join(deficits)})")
    print(f"Submission: {state.dep.submission_path}")
    print(f"Token spend: {state.token_spend}")
    print(f"Final state written to {state_path}")
    print(f"Reports:     {paths.reports}/")
    if interrupted:
        return 130
    if run_failed:
        return 1
    return 0 if ml_run_succeeded(state) else 1


def cmd_flow_plot(args: argparse.Namespace) -> int:
    """Write an HTML visualization of CrispDMFlow."""
    from maads.flow.crisp_dm_flow import CrispDMFlow
    from maads.config import load_case_config
    from maads.paths import resolve_path

    config_path = resolve_path("configs/titanic.yaml")
    config = load_case_config(config_path)
    state = CrispDMState.from_config(config)
    flow = CrispDMFlow(state, Path("artifacts/plot"))
    flow.plot(args.output)
    print(f"Flow visualization saved to {args.output}.html")
    return 0


def cmd_artifacts_render(args: argparse.Namespace) -> int:
    from maads.artifacts_render import render_trace_views

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = resolve_path(run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        return 1
    written = render_trace_views(run_dir)
    print(f"Wrote {len(written)} render file(s) under {run_dir / 'trace'}")
    return 0


def cmd_artifacts_backfill_timing(args: argparse.Namespace) -> int:
    from maads.artifacts_timing import backfill_all_runs, backfill_run_timing

    if args.run:
        run_dir = Path(args.run)
        if not run_dir.is_absolute():
            run_dir = resolve_path(run_dir)
        if not run_dir.is_dir():
            print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
            return 1
        results = [backfill_run_timing(run_dir, dry_run=args.dry_run)]
    else:
        artifact_root = resolve_path(args.artifact_dir)
        results = backfill_all_runs(artifact_root, dry_run=args.dry_run)

    updated = [r for r in results if r["changed"]]
    print(
        f"{'Would update' if args.dry_run else 'Updated'} "
        f"{len(updated)} of {len(results)} run(s)."
    )
    for row in updated:
        timing = row["timing"]
        dur = timing.get("duration_ms")
        dur_s = f"{dur // 1000 // 60}m {(dur // 1000) % 60}s" if dur else "n/a"
        print(f"  {row['run_id']}: {dur_s} ({', '.join(row['changed'])})")
    return 0


def cmd_data_download(args: argparse.Namespace) -> int:
    if args.case:
        download_case_data(args.case,
                           data_dir=Path(args.out_dir) if args.out_dir else None)
    else:
        out = Path(args.out_dir) if args.out_dir else Path("data") / args.competition
        download_kaggle_competition(args.competition, out)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Start the trace monitoring dashboard."""
    try:
        from maads.dashboard import run_dashboard
    except ImportError:
        print(
            "ERROR: dashboard dependencies not installed. "
            'Run: pip install -e ".[dashboard]"',
            file=sys.stderr,
        )
        return 1

    artifact_root = resolve_path(args.artifact_dir)
    static_dir = args.static_dir
    if static_dir is None:
        default_static = resolve_path("dashboard") / "dist"
        if default_static.is_dir():
            static_dir = default_static
    elif not static_dir.is_absolute():
        static_dir = resolve_path(static_dir)

    print(f"Artifact root: {artifact_root.resolve()}")
    print(f"Dashboard:     http://{args.host}:{args.port}/")
    print(f"API docs:      http://{args.host}:{args.port}/api/docs")
    if static_dir:
        print(f"Frontend:      {static_dir}")
    else:
        print("Frontend:      not built — use `cd dashboard && npm run dev` for dev UI")

    run_dashboard(
        artifact_dir=artifact_root,
        host=args.host,
        port=args.port,
        static_dir=static_dir,
        open_browser=not args.no_open,
        case_id=args.case,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
