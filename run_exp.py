import OPTAgent
import argparse
import logging
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the OPTAgent experiment.")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory where the data is located"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of steps to run the experiment"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for the model"
    )
    parser.add_argument(
        "--draft_refine", type=str, default="refine", choices=["draft", "refine"], help="Draft or Refine"
    )
    parser.add_argument(
    "--base_model",
    type=str,
    choices=["gpt-4o-2024-08-06", "o3-mini", "claude-3-7-sonnet-20250219","grok-3","gpt-4.1-2025-04-14", "gemini-2.0-flash", "Qwen/Qwen2.5-72B-Instruct","claude-3-5-sonnet-20241022","DeepSeek-R1-BF16","Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-7B-Instruct"],
    default="gpt-4o-2024-08-06",
    help="Please specify the base model."
)
    parser.add_argument('--use_memory', type=lambda x: x.lower() == 'true', default=True,
                      help='Boolean flag for using memory')
    parser.add_argument('--task_field', type=str, default="ML", choices=["ML", "NP"],
                      help='sloving ML task or NP task')
    # Parse arguments
    args = parser.parse_args()

    # Set up logger
    logger = logging.getLogger("OPTAgent")
    logging.basicConfig(level=logging.INFO)
    logger.info(f'Starting run with data directory: "{args.data_dir}"')

    # Run the experiment
    task_type = os.path.basename(args.data_dir)
    if task_type == "":
        task_type = os.path.basename(os.path.dirname(args.data_dir))
    print(f"use_memory: {args.use_memory}")
    print(f"task_field: {args.task_field}")
    print(f"draft_refine: {args.draft_refine}")
    print(f"temperature: {args.temperature}")
    exp = OPTAgent.Experiment(data_dir=args.data_dir,task_type=task_type, base_model=args.base_model, use_memory=args.use_memory,task_field=args.task_field,draft_refine=args.draft_refine,temperature=args.temperature,steps=args.steps)

    if args.task_field == "ML":
        best_solution = exp.run_ML(steps=args.steps)
        # Output the best solution's validation metric and code
        if best_solution:
            print(f"Best solution has validation metric: {best_solution.valid_metric}")
        else:
            print("No valid solution found.")
        # print(f"Best solution code: {best_solution.code}")
    else:
        good_node_metric_list, best_node_metric_list = exp.run_NP(steps=args.steps)
        print(f"Good node metric list: {good_node_metric_list}, average: {sum(good_node_metric_list)/len(good_node_metric_list)}")
        print(f"Best node metric list: {best_node_metric_list}, average: {sum(best_node_metric_list)/len(best_node_metric_list)}")

if __name__ == "__main__":
    main()