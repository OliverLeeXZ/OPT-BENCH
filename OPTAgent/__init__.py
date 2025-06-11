from dataclasses import dataclass
import os
from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    _load_cfg,
    prep_cfg,
    extract_markdown_content,
    extract_markdown_content_NP,
)
from .journal2report import journal2report
import logging
import json
@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:

    def __init__(self, data_dir: str, task_type: str| None = None, base_model: str = "gpt4o", use_memory: bool = True, task_field: str = "ML",draft_refine: str = "refine",temperature: float = 0.5, steps: int = 10):
        """Initialize ML experiment to run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        self.base_model = base_model
        self.use_memory: bool = use_memory
        self.task_type = task_type
        self.task_type_md = task_type + ".md"
        self.task_field = task_field
        self.draft_refine = draft_refine
        self.temperature = temperature
        self.steps= steps
        self.cfg = prep_cfg(_cfg, task_name=task_type, model_name=base_model, use_memory=self.use_memory, draft_refine=self.draft_refine, temperature=self.temperature,steps=self.steps)
        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    
    def run_ML(self, steps: int) -> Solution:
        self.journal = Journal()
        self.task_info = extract_markdown_content(os.path.join(self.cfg.data_dir,self.task_type_md))
        self.agent = Agent(
            cfg=self.cfg,
            journal=self.journal,
            task_type= self.task_type,
            base_model=self.base_model,
            use_memory=self.use_memory,
            task_field=self.task_field,
            draft_refine=self.draft_refine,
            temperature=self.temperature,
            NP_question=None,
            task_info=self.task_info
        )
        for _i in range(steps):
            print("Step", _i + 1)
            if self.use_memory:
                self.agent.step(exec_callback=self.interpreter.run)
            else:
                self.agent.random_gen(exec_callback=self.interpreter.run, _i=_i)
            print(f"Step {_i+1} complete.")
        save_run(self.cfg, self.journal, use_memory=self.use_memory, task_field=self.task_field)
        self.interpreter.cleanup_session()
        return self.get_best_solution()
        
    def run_NP(self, steps: int) -> Solution:
        self.journal = Journal()
        self.task_info_NP = extract_markdown_content_NP(os.path.join(self.cfg.data_dir,self.task_type_md))
        self.question_json = os.path.join(self.cfg.data_dir, "question.json")
        with open(self.question_json, 'r', encoding='utf-8') as f:
            self.question_data = json.load(f)
        
        results = {}  # Dictionary to store results for each question
        good_node_metric_list = []  # List to store good node metrics
        best_node_metric_list = []  # List to store best node metrics
        questions = self.question_data["questions"]
        if "big_small" in self.question_data:
            self.big_small = self.question_data["big_small"]
        else:
            self.big_small = "big"
        for question_id, question_data in questions.items():
            print(f"\nProcessing {question_id}...")
            
            # Create a subdirectory for this question
            question_log_dir = self.cfg.log_dir / question_id
            question_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a copy of config with updated log directory
            question_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg))
            question_cfg.log_dir = question_log_dir
            
            self.agent = Agent(
                cfg=self.cfg,
                journal=self.journal,
                task_type=self.task_type,
                base_model=self.base_model,
                use_memory=self.use_memory,
                task_field=self.task_field,
                draft_refine=self.draft_refine,
                temperature=self.temperature,
                NP_question=question_data,
                task_info=self.task_info_NP
            )
            
            for _i in range(steps):
                print(f"{question_id} - Step {_i + 1}")
                if self.use_memory:
                    self.agent.step_np()
                else:
                    self.agent.random_gen_np(_i=_i)
                    
            print(f"{question_id} - Step {_i+1} complete.")
            
            # Save results for this question
            save_run(question_cfg, self.journal, use_memory=self.use_memory, task_field=self.task_field, big_small=self.big_small)
            good_node_metric, best_node_metric = self.get_np_solution()
            
            # Store results in both formats
            results[question_id] = {
                'good_node_metric': good_node_metric,
                'best_node_metric': best_node_metric
            }
            good_node_metric_list.append(good_node_metric)
            best_node_metric_list.append(best_node_metric)
            
            # 计算平均值
            good_node_avg = sum(good_node_metric_list)/len(good_node_metric_list)
            best_node_avg = sum(best_node_metric_list)/len(best_node_metric_list)
            
            # 打印到控制台
            print(f"Good node metric list: {good_node_metric_list}, average: {good_node_avg}")
            print(f"Best node metric list: {best_node_metric_list}, average: {best_node_avg}")
            
            # 写入log文件
            log_path = self.cfg.log_dir / "metrics_log.txt"
            with open(log_path, 'a') as f:
                f.write(f"big_small: {self.big_small}\n")
                f.write(f"\nQuestion {question_id} completed:\n")
                f.write(f"Good node metric list: {good_node_metric_list}\n")
                f.write(f"Good node average: {good_node_avg}\n")
                f.write(f"Best node metric list: {best_node_metric_list}\n")
                f.write(f"Best node average: {best_node_avg}\n")
                f.write("-" * 50 + "\n")
            
            # Reset journal for next question
            self.journal = Journal()
        
        self.interpreter.cleanup_session()
        
        # Save overall results summary
        summary_path = self.cfg.log_dir / "results_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return good_node_metric_list, best_node_metric_list

    def get_np_solution(self):
        good_node = self.journal.good_nodes[0] if self.journal.good_nodes else None
        best_node = self.journal.get_best_node(only_good=True, big_small=self.big_small)
        if good_node and best_node:
            return good_node.metric, best_node.metric
        elif not good_node :
            return 0,0
        
    def generate_report(self):
        if self.cfg.generate_report:
            print("Generating final report from journal...")
            report = journal2report(self.journal, self.task_desc, self.cfg.report)
            print(report)
            report_file_path = self.cfg.log_dir / "report.md"
            with open(report_file_path, "w") as f:
                f.write(report)
            print("Report written to file:", report_file_path)
        
    def get_best_solution(self) -> Solution:
        init_best_node = logging.getLogger("init_best_node")
        init_best_node.setLevel(logging.INFO)
        handler = logging.StreamHandler()  # You can also log to a file if needed
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        init_best_node.addHandler(handler)

        # Get the first node (init_node) and the best node (best_node) from the journal
        init_node = self.journal.good_nodes[0] if self.journal.good_nodes else None
        best_node = self.journal.get_best_node_ML(only_good=True)
        if best_node:
            if self.task_field == "ML":
                return Solution(code=best_node.code, valid_metric=best_node.metric.value)
            else:
                return Solution(code=best_node.code, valid_metric=best_node.metric)
        else:
            return None