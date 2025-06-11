import logging
import random
from typing import Any, Callable, cast
import os
import humanize
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code, extract_code_block
from .utils.config import  extract_markdown_content, mergecode
import json
logger = logging.getLogger("agent")
ExecCallbackType = Callable[[str, bool], ExecutionResult]
review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)
def load_prompt_ML_template(prompt_type: str) -> dict:
    # load json
    json_path =  'OPTAgent/utils/prompt_ML.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        prompt_list = json.load(f)
    # find type
    for item in prompt_list:
        if item['type'] == prompt_type:
            prompt = item['prompt']
            break
    else:
        raise ValueError(f"Prompt type {prompt_type} not found in prompt_ML.json")
    return prompt

def load_prompt_NP_template(prompt_type: str) -> dict:
    # load json
    json_path =  'OPTAgent/utils/prompt_NP.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        prompt_list = json.load(f)
    # find type
    for item in prompt_list:
        if item['type'] == prompt_type:
            prompt = item['prompt']
            break
    else:
        raise ValueError(f"Prompt type {prompt_type} not found in prompt_ML.json")
    return prompt

class Agent:
    def __init__(
        self,
        cfg: Config,
        journal: Journal,
        task_type: str,
        base_model: str,
        use_memory: bool,
        task_field: str,
        draft_refine: str,
        temperature: float,
        NP_question: Any,
        task_info: dict,
    ):
        super().__init__()
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.task_type = task_type
        self.task_info = task_info
        self.base_model = base_model
        self.use_memory = use_memory
        if task_field == "NP":
            validation_py = os.path.join(self.cfg.data_dir, "validation.py")
            with open(validation_py, "r", encoding="utf-8") as f:
                self.validation_code = f.read()
        self.draft_refine = draft_refine
        self.temperature = temperature
        self.NP_question = NP_question
    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        # initial drafting
        search_cfg = self.acfg.search
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] begin with the init solution")
            return None
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")
            
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None
        
        good_node_improve = self.journal.good_nodes[-1]
        logger.debug("[search policy] improve node selected")
        return good_node_improve
    
    def search_policy_np(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        # initial drafting
        search_cfg = self.acfg.search
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] np begin with the init solution")
            return None
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] np debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")
            
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] np drafting new node (no good nodes)")
            return None
        
        good_node_improve = self.journal.good_nodes[-1]
        logger.debug("[search policy] improve node selected")
        return good_node_improve
    
    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "catboost",
            "optuna"
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            f"All data is already available in the ./input directory."
            f'You can also use the ./working directory to store any temporary files that your code needs to create.'
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation >1 :
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}
    @property
    def _prompt_impl_guideline_draft(self):
        impl_guideline = [
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            f"All data is already available in the ./input directory."
            f'You can also use the ./working directory to store any temporary files that your code needs to create.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!'
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation >1 :
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        #implemente the solution in a single code block
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution for model, optimizer and hyperparameters selection in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Note that the code block should be a complete Python program."
        )
        }
    @property
    def _prompt_instructions_draft(self):
        return {
            "Solution sketch guideline": [
                "The initial solution design should be simple, efficient, and avoid overfitting, with minimal iterations.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do Exploratory Data Analysis (EDA).",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                "Note that the training dataset should be shuffled before splitting it into training and validation sets, and the random seed (state) should be fixed."
            ]
        }
    @property
    def _prompt_instructions_improve(self):
        return {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "Don't suggest to do EDA (Exploratory Data Analysis)",
                "Ensure function parameters match official documentation, check for accuracy, compatibility, and any deprecated or renamed parameters. Refer to the latest examples if needed.",
                "Note that only modify the model, optimizer, hyperparameters, and adjust feature engineering."
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
            ],
        }
    @property
    def _prompt_instructions_debug(self):
        return {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA (Exploratory Data Analysis)",
                "Ensure function parameters match official documentation, check for accuracy, compatibility, and any deprecated or renamed parameters. Refer to the latest examples if needed.",
                "If the previous buggy solution was due to time limitations, focus on reducing the time consumption of the code instead of fixing the bug. For example, simplify the model's hyperparameters, reduce the number of iterations, or change from K-Fold cross-validation to a Single Train-Test Split.",
                "Take the Memory section into consideration when proposing the improvement.",
            ],
        }
    
        
    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.base_model,
                temperature=self.acfg.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore
    
    def result_query(self, prompt, retries=3) -> tuple[str, str]:
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.base_model,
                temperature=self.acfg.code.temp,
            )
            if completion_text:
                return completion_text
        print("Final plan attempt failed, giving up...")
        return  completion_text  # type: ignore
    
    def _draft(self) -> Node:
        prompt = load_prompt_ML_template("draft")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Evaluation metric"] = prompt["Evaluation metric"].format(metric=self.task_info["metric"])
        prompt["Training set format"] = prompt["Training set format"].format(dataset_description=self.task_info["dataset_description"])
        prompt["Submission format"] = prompt["Submission format"].format(submission_format=self.task_info["submission_format"])
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_instructions_draft
        prompt["Instructions"] |= self._prompt_impl_guideline_draft
        prompt["Instructions"] |= self._prompt_environment
        
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code )
    
    def _draft_NP(self) -> Node:
        #adding the template code to the prompt
        prompt = load_prompt_NP_template("draft")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Example Input and Output"] = prompt["Example Input and Output"].format(example_input=self.task_info["example_input"],example_output=self.task_info["example_output"])   
        prompt["Submission Format"] = prompt["Submission Format"].format(submission_format=self.task_info["submission_format"])
        prompt["Question"] = prompt["Question"].format(task_type=self.task_type,question=self.NP_question)
        code = self.result_query(prompt)
        return Node(code=code)
        
    def _improve_wom(self, parent_node: Node) -> Node:
        prompt = load_prompt_ML_template("improve_wom")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Training set format"] = prompt["Training set format"].format(dataset_description=self.task_info["dataset_description"])
        prompt["Previous solution"]["Code"] = prompt["Previous solution"]["Code"].format(code_template=self.task_info["code_template"])
        
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_instructions_improve
        prompt["Instructions"] |= self._prompt_impl_guideline
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )
        

    def _improve(self, parent_node: Node) -> Node:
        #adding the template code to the prompt
        prompt = load_prompt_ML_template("improve")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Training set format"] = prompt["Training set format"].format(dataset_description=self.task_info["dataset_description"])
        prompt["History information"] = prompt["History information"].format(history_information=self.journal.generate_summary())
        prompt["Previous solution"]["Code"] = prompt["Previous solution"]["Code"].format(code_template=wrap_code(parent_node.code))
        
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_instructions_improve
        prompt["Instructions"] |= self._prompt_impl_guideline
        
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )
        
    def _improve_NP(self, parent_node: Node) -> Node:
        #adding the template code to the prompt
        prompt = load_prompt_NP_template("improve")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Example Input and Output"] = prompt["Example Input and Output"].format(example_input=self.task_info["example_input"],example_output=self.task_info["example_output"]) 
        prompt["Submission Format"] = prompt["Submission Format"].format(submission_format=self.task_info["submission_format"])
        prompt["Question"] = prompt["Question"].format(task_type=self.task_type,question=self.NP_question)
        prompt["History information"] = prompt["History information"].format(history_information=self.journal.generate_summary_np())
        
        code = self.result_query(prompt)
        return Node(code=code, parent=parent_node)
    
    def _improve_NP_wom(self, parent_node: Node) -> Node:
        #adding the template code to the prompt
        prompt = load_prompt_NP_template("improve_wom")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Example Input and Output"] = prompt["Example Input and Output"].format(example_input=self.task_info["example_input"],example_output=self.task_info["example_output"]) 
        prompt["Submission Format"] = prompt["Submission Format"].format(submission_format=self.task_info["submission_format"])
        prompt["Question"] = prompt["Question"].format(task_type=self.task_type,question=self.NP_question)
        code = self.result_query(prompt)
        return Node(code=code, parent=parent_node)

    def _debug(self, parent_node: Node) -> Node:
        #adding debuggy log and error message to the prompt
        prompt = load_prompt_ML_template("debug")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Previous (buggy) implementation"] = prompt["Previous (buggy) implementation"].format(buggy_code=wrap_code(parent_node.code))
        prompt["Execution output"] = prompt["Execution output"].format(execution_output=wrap_code(parent_node.term_out, lang=""))
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_instructions_debug
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def _debug_NP(self, parent_node: Node) -> Node:
        #adding debuggy log and error message to the prompt
        prompt = load_prompt_NP_template("debug")
        prompt["Introduction"] = prompt["Introduction"].format(task_type=self.task_type)
        prompt["Task description"] = prompt["Task description"].format(description=self.task_info["description"])
        prompt["Example Input and Output"] = prompt["Example Input and Output"].format(example_input=self.task_info["example_input"],example_output=self.task_info["example_output"]) 
        prompt["Submission Format"] = prompt["Submission Format"].format(submission_format=self.task_info["submission_format"])
        prompt["Question"] = prompt["Question"].format(task_type=self.task_type,question=self.NP_question)
        prompt["Previous buggy information"] = prompt["Previous buggy information"].format(previous_buggy_information=parent_node.analysis)
        code = self.result_query(prompt)
        return Node(code=code, parent=parent_node)

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def random_gen(self, exec_callback: ExecCallbackType, _i: int):
        if _i == 0:
            result_node =  Node(plan='The first solution needs to be improved.',code=extract_code_block(self.task_info["code_template"]))
            print('Begin with the initial solution')
        else:
            result_node = self._improve_wom(self.journal.nodes[0])
            print('Genrerating the improving solution randomly')

        result_node.full_solution = result_node.code
        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)
        
    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            if self.draft_refine == "refine":
                result_node =  Node(plan='The first solution needs to be improved.',code=extract_code_block(self.task_info["code_template"]))
                print('Begin with the initial solution')
            elif self.draft_refine == "draft":
                result_node = self._draft()
                print('Drafting the initial solution')
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
            print('Debugging the previous solution')
        else:
            result_node = self._improve(parent_node)
            print('Improving the previous solution')

        result_node.full_solution = result_node.code
        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

    def step_np(self):
        parent_node = self.search_policy_np()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft_NP()
            print('drafting the initial np solution')
        elif parent_node.is_buggy:
            result_node = self._debug_NP(parent_node)
            print('Debugging the previous np solution')
        else:
            result_node = self._improve_NP(parent_node)
            print('Improving the previous np solution')

        self.validate_NP(result_node)
        self.journal.append(result_node)

    def random_gen_np(self, _i: int):
        if _i == 0:
            result_node =  self._draft_NP()
            print('Begin with the initial solution')
        else:
            result_node = self._improve_NP_wom(self.journal.nodes[0])
            print('Genrerating the improving solution randomly')
        self.validate_NP(result_node)
        self.journal.append(result_node)

    def validate_NP(self, node: Node) -> Node:
        # print("node.code:", node.code)
        # exec validation.py
        local_vars = {}
        exec(self.validation_code, {}, local_vars)
        
        validate_func = local_vars.get("validation")
        if validate_func is None:
            raise RuntimeError("validation.py must define a 'validation' function")
        result, metric, info = validate_func(self.NP_question, node.code)
        print("result:", result, "metric:", metric, "info:", info)
        
        node.is_buggy = result
        node.metric = metric
        node.analysis = info
        
        
        
    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            ),
            "Task description": self.task_info["description"],
            "Evaluation metric": self.task_info["metric"],
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                # model=self.base_model,
                model='gpt-4o-2024-08-06',
                temperature=self.acfg.feedback.temp,
            ),
        )
            
        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )

