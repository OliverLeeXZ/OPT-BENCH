import atexit
import logging
import shutil
import sys
import os
from . import backend

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg

class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)

def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution Path")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree

def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            # support for multiple markers; atm only "best" is supported
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            if marker_str:
                s = f"{indent}● {node.metric.value:.3f} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {node.metric.value:.3f} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str

def run():
    cfg = load_cfg()
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, "INFO", logging.INFO), format=log_format, handlers=[]
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    
    task_type = sys.argv[1]  # Pass task_type as a command-line argument
    # starting run experiment
    logger = logging.getLogger("OPTAgent")
    logger.info(f'Starting run "{cfg.exp_name}"')
    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)
    
    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        task_type=task_type,
    )
    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
    )

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def generate_live():
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"), prog, status
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]OPTAgent is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    with Live(
        generate_live(),
        refresh_per_second=16,
        screen=True,
    ) as live:
        while global_step < cfg.agent.steps:
            agent.step(exec_callback=exec_callback)
            save_run(cfg, journal)
            global_step = len(journal)
            live.update(generate_live())
    # output the solution tree
    log_tree_path = cfg.log_dir / "solution_tree.log"
    logger = logging.getLogger("solution_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_tree_path, mode='w')  
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # write solution tree log
    logger.info(journal_to_string_tree(journal))
    interpreter.cleanup_session()
    if cfg.generate_report:
        print("Generating final report from journal...")
        report = journal2report(journal, task_desc, cfg.report)
        print(report)
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    run()
