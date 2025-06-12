"""configuration and setup utils"""

from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, cast

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging
import json
from . import tree_export
from . import copytree, preproc_data, serialize
import re
import black
import datetime

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("OPTAgent")
logger.setLevel(logging.WARNING)

""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class AgentConfig:
    steps: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig

    search: SearchConfig


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None
    task_type: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config, task_name: str, model_name: str, use_memory: bool, draft_refine: str, temperature: float, steps:int):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)
    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or task_name
    # cfg.exp_name = f"{model_name}-{cfg.exp_name}"
    cfg.exp_name = f"{ind}-{model_name}-{cfg.exp_name}-{use_memory}-{draft_refine}-{temperature}-{steps}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()
    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )
        
    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval
    if cfg.task_type is not None:
        task_desc["Task type"] = cfg.task_type
    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    # Ensure workspace directories exist
    input_dir = cfg.workspace_dir / "input"
    working_dir = cfg.workspace_dir / "working"

    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the directory is empty or the data directory is already copied
    if not any(input_dir.iterdir()):  # If the input directory is empty
        copytree(cfg.data_dir, input_dir, use_symlinks=not cfg.copy_data)
    
    if cfg.preprocess_data:
        preproc_data(input_dir)

def output_file_or_placeholder(file: Path):
    if file.exists():
        if file.suffix != ".json":
            return file.read_text()
        else:
            return json.dumps(json.loads(file.read_text()), indent=4)
    else:
        return f"File not found."
    
def concat_logs(chrono_log: Path, best_node: Path, journal: Path):
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the OPTAgent run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content

def extract_markdown_content(md_file_path: str) -> dict:
    # Read the content of the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Define a dictionary to store the task description information
    task_des = {}

    # Match the content of each section (starting with '##' headers)
    sections = {
        "description": r"## Description\n(.*?)## Metric",
        "metric": r"## Metric\n(.*?)## Submission Format",
        "submission_format": r"## Submission Format\n(.*?)## Dataset Description",
        "dataset_description": r"## Dataset Description\n(.*?)## Code Template", # Modified regular expression
        "code_template": r"## Code Template\n(.*?)(##|$)",
    }

    # Extract and store the corresponding content for each section
    for key, pattern in sections.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            task_des[key] = match.group(1).strip()
        else:
            task_des[key] = None

    return task_des

def extract_markdown_content_NP(md_file_path: str) -> dict:
    # Read the content of the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Define a dictionary to store the task description information
    task_des = {}

    # Match the content of each section (starting with '##' headers)
    sections = {
        "description": r"## Description\n(.*?)## Submission Format",
        "submission_format": r"## Submission Format\n(.*?)## Example Input",
        "example_input": r"## Example Input\n(.*?)## Example Output",
        "example_output": r"## Example Output\n(.*?)(##|$)",
    }

    # Extract and store the corresponding content for each section
    for key, pattern in sections.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            task_des[key] = match.group(1).strip()
        else:
            task_des[key] = None

    return task_des

def mergecode(code_template:str, model_code:str)-> str:
    # replace the placeholder with the actual code
    full_code = code_template.replace("<model and optimizer>", model_code)

    # formatting the code using black
    formatted_code = black.format_str(full_code, mode=black.FileMode())
    
    return formatted_code
#adding init node metric and best metric difference
def save_run(cfg: Config, journal, use_memory:bool, task_field="ML", big_small="big"):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # save the best found solution
    best_node = journal.get_best_node(only_good=True)
    if task_field == "ML":
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            if best_node is not None:
                f.write(best_node.code)
            else:
                f.write("No solution found.")
    all_log_path = cfg.log_dir / "all_nodes_info.log"
    # === save all nodes to log ===
    all_nodes_info = journal.nodes
    with open(all_log_path, "a") as all_file:
        all_file.write("\n" + "="*50 + "\n")
        all_file.write(f"Run timestamp: {datetime.datetime.now()}\n")
        all_file.write("All node Log:\n") 
        for node in all_nodes_info:
            all_file.write(f"Node ID: {node.id}\n")
            all_file.write(f"Analysis: {node.analysis}\n")
            if task_field == "ML":
                all_file.write(f"Execution Info: {node.exc_info}\n")
            if not node.is_buggy:
                all_file.write(f"Metric: {node.metric}\n")
            all_file.write(f"Code: {node.code}\n")
            all_file.write("-" * 50 + "\n")
        
    # === save init and best node to file ===
    init_best_node = logging.getLogger("init_best_node")
    init_best_node.setLevel(logging.INFO)
    handler = logging.StreamHandler()  # You can also log to a file if needed
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    init_best_node.addHandler(handler)
    # Get the first node (init_node) and the best node (best_node) from the journal
    init_node = journal.good_nodes[0] if journal.good_nodes else None
    best_node = journal.get_best_node(only_good=True)

    if init_node is not None:
        # Log the information for init_node
        init_best_node.info(f"Initial node info:\Metric: {init_node.metric}, Analysis: {init_node.analysis}")
    else:
        init_best_node.info("No initial node found.")
    if best_node is not None:
        # Log the information for best_node
        init_best_node.info(f"Initial node info:\Metric: {best_node.metric}, Analysis: {best_node.analysis}")
    else:
        init_best_node.info("No best node found.")

    log_tree_path = cfg.log_dir / "solution_tree.log"
    RED = '\033'
    RESET = '\033'  # Reset to the default color

    with open(log_tree_path, "a") as log_file:
        log_file.write("\n" + "="*50 + "\n")
        log_file.write(f"Run timestamp: {datetime.datetime.now()}\n")
        log_file.write("Solution path Log:\n")
        if use_memory:
            log_file.write("Using memory:\n")
        else:
            log_file.write("Not using memory:\n")
        if task_field == "ML":
            log_file.write("ML task\n")
        elif task_field == "NP":
            log_file.write("NP task\n")
        for idx, node in enumerate(journal.nodes):
            status = "buggy" if node.is_buggy else f"good, metric: {node.metric}"

            # If it's the best node, apply red color to the terminal output
            if task_field == "ML":
                if node == journal.get_best_node_ML(only_good=True):
                    status = f"{RED}best node, metric: {node.metric}{RESET}"
            elif task_field == "NP":
                if node == journal.get_best_node(only_good=True, big_small=big_small):
                    status = f"{RED}best node, metric: {node.metric}{RESET}"
            log_message = f"Step: {node.step}, Node ID: {node.id}, Status: {status}"
            
            # Log to the terminal (colored text if it's the best node)
            logger.info(log_message)  
            
            # Write the plain text log to the log file (no color)
            log_file.write(f"Step: {node.step}, Node ID: {node.id}, Status: {status}\n")  # Log without ANSI codes

            if (idx + 1) % 5 == 0:
                sep_line = "-" * 40 + "\n"
                log_file.write(sep_line)
                logger.info(sep_line.strip())
        
        log_file.write("End of All node Log.\n") 