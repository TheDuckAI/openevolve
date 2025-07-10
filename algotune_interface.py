import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from AlgoTuneTasks.factory import TaskFactory


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", type=str, required=True, help="name of the AlgoTune task")
    args = parser.parse_args()

    openevolve_config_dir = f"./examples/{args.task}"

    os.makedirs(openevolve_config_dir, exist_ok=True)

    task_dir = f"./AlgoTune/AlgoTuneTasks/{args.task}"
    task_path = os.path.join(task_dir, f"{args.task}.py")
    with open(task_path) as f:
        source = f.read()

    with open(os.path.join(openevolve_config_dir, "initial_program.py"), "w") as f:
        f.write(f"# EVOLVE-BLOCK-START\n{source}\n# EVOLVE-BLOCK-END")


if __name__ == "__main__":
    main()

task: Task = TaskFactory("base64_encoding")
# print(task.generate_problem(n=1))

# initial_program.py is just the algotune task source code
# the evaluator is just importing the task and verifying that it is right compared to the original problem

# TODO: for prompt, load specific description into it
