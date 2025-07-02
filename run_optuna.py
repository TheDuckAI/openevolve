import argparse
import asyncio
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial

import optuna

from openevolve import OpenEvolve
from openevolve.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("initial_program", help="Path to the initial program file")

    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results")

    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    parser.add_argument("--n_trials", type=int, default=1, help="number of trials to run")

    parser.add_argument("--n_jobs", type=int, default=1, help="number of parallel jobs to run")

    parser.add_argument(
        "--timeout", type=float, default=None, help="global timeout in seconds for optimization"
    )

    return parser.parse_args()


def objective(trial: optuna.Trial, args: argparse.Namespace):
    # Create config object with command-line overrides
    # Load base config from file or defaults
    config = load_config(args.config)

    # Apply command-line overrides
    if args.api_base:
        config.llm.api_base = args.api_base
        print(f"Using API base: {config.llm.api_base}")

    if args.primary_model:
        config.llm.primary_model = args.primary_model
        print(f"Using primary model: {config.llm.primary_model}")

    if args.secondary_model:
        config.llm.secondary_model = args.secondary_model
        print(f"Using secondary model: {config.llm.secondary_model}")

    # Set up config ranges

    # general config
    # config.max_iterations = trial.suggest_int("max_iterations", low=10, high=100)
    # config.random_seed = trial.suggest_int("random_seed", low=0, high=10_000)

    # llm config
    # config.llm.temperature = trial.suggest_float("temperature", low=0.3, high=1.0)
    # config.llm.top_p = trial.suggest_float("top_p", low=0.8, high=1.0)
    # config.llm.max_tokens = trial.suggest_float("max_tokens", low=4096, high=32768, log=True)
    # TODO: add arbitrary number of models here
    # config.llm.primary_model = trial.suggest_categorical(
    #     "primary_model",
    #     [
    #         "google/gemini-2.0-flash-001",
    #         "google/gemini-2.5-flash",
    #         "openai/gpt-4.1-mini",
    #     ],
    # )
    # config.llm.primary_model_weight = trial.suggest_float("primary_model_weight", low=0.6, high=0.9)
    # config.llm.secondary_model = trial.suggest_categorical(
    #     "secondary_model",
    #     [
    #         "anthropic/claude-3.7-sonnet",
    #         "anthropic/claude-4-sonnet",
    #         "openai/gpt-4.1",
    #     ],
    # )
    # config.llm.secondary_model_weight = 1 - config.llm.primary_model_weight
    # set for experiment 2:
    # config.llm.primary_model = "google/gemini-2.0-flash-001"
    # config.llm.primary_model_weight = 1.0
    # config.llm.secondary_model = "anthropic/claude-3.7-sonnet"
    # config.llm.secondary_model_weight = 0.0

    # prompt config
    config.prompt.num_top_programs = trial.suggest_int("num_top_programs", low=3, high=10)
    config.prompt.num_diverse_programs = trial.suggest_int("num_diverse_programs", low=2, high=5)
    # config.prompt.include_artifacts = trial.suggest_categorical("include_artifacts", [False, True])

    # database config
    config.database.population_size = trial.suggest_int(
        "population_size", low=40, high=150, step=10
    )
    config.database.archive_size = trial.suggest_int("archive_size", low=10, high=60, step=5)
    config.database.num_islands = trial.suggest_int("num_islands", low=3, high=10)

    # config.database.elite_selection_ratio = trial.suggest_float(
    #     "elite_selection_ratio", low=0.1, high=0.5
    # )
    # config.database.exploration_ratio = trial.suggest_float("exploration_ratio", low=0.1, high=0.9)
    # config.database.exploitation_ratio = 1 - config.database.exploration_ratio

    # feature_dimensions = ["complexity", "diversity", "score"]
    # all_subsets = []
    # for k in range(1, len(feature_dimensions) + 1):
    #     all_subsets += tuple(combinations(feature_dimensions, k))
    # config.database.feature_dimensions = trial.suggest_categorical(
    #     "feature_dimensions", all_subsets
    # )

    config.database.migration_interval = trial.suggest_int(
        "migration_interval", low=20, high=100, step=10
    )
    config.database.migration_rate = trial.suggest_float("migration_rate", low=0.05, high=0.25)

    # evaluator config
    # TODO: add cascade_threshold
    # config.evaluator.use_llm_feedback = trial.suggest_categorical("use_llm_feedback", [False, True])
    # config.evaluator.llm_feedback_weight = trial.suggest_float(
    #     "llm_feedback_weight", low=0.05, high=0.3
    # )

    # evolution config
    # config.diff_based_evolution = trial.suggest_categorical("diff_based_evolution", [False, True])

    # Initialize OpenEvolve
    try:
        openevolve = OpenEvolve(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config=config,
            output_dir=os.path.join(args.output, f"trial{trial.number}"),
        )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                raise ValueError(f"Checkpoint directory '{args.checkpoint}' not found")
            print(f"Loading checkpoint from {args.checkpoint}")
            openevolve.database.load(args.checkpoint)
            print(
                f"Checkpoint loaded successfully (iteration {openevolve.database.last_iteration})"
            )

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = asyncio.run(
            openevolve.run(iterations=args.iterations, target_score=args.target_score, trial=trial)
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(openevolve.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        print("\nEvolution complete!")
        print("Best program metrics:")
        for name, value in best_program.metrics.items():
            # Handle mixed types: format numbers as floats, others as strings
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return best_program.metrics["combined_score"]

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def load_params_from_config(args):
    config = load_config(args.config)
    defaults = {
        # "max_iterations": config.max_iterations,
        # "random_seed": config.random_seed,
        "temperature": config.llm.temperature,
        # "top_p": config.llm.top_p,
        # "max_tokens": config.llm.max_tokens,
        "primary_model": config.llm.primary_model,
        "primary_model_weight": config.llm.primary_model_weight,
        "secondary_model": config.llm.secondary_model,
        "num_top_programs": config.prompt.num_top_programs,
        "num_diverse_programs": config.prompt.num_diverse_programs,
        # "include_artifacts": config.prompt.include_artifacts,
        "population_size": config.database.population_size,
        "archive_size": config.database.archive_size,
        "num_islands": config.database.num_islands,
        "elite_selection_ratio": config.database.elite_selection_ratio,
        "exploration_ratio": config.database.exploration_ratio,
        # "exploitation_ratio": config.database.exploitation_ratio,
        # "feature_dimensions": tuple(config.database.feature_dimensions),
        "migration_interval": config.database.migration_interval,
        "migration_rate": config.database.migration_rate,
        # "use_llm_feedback": config.evaluator.use_llm_feedback,
        # "llm_feedback_weight": config.evaluator.llm_feedback_weight,
        # "diff_based_evolution": config.diff_based_evolution,
    }
    return defaults


# TODO: run with multiprocessing instead of threading here. need to switch to async DB like postgres
# TODO: make single DB for all experiments


def main():
    args = parse_args()

    # Check if files exist
    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    os.makedirs(args.output, exist_ok=True)

    study = optuna.create_study(
        storage=f"sqlite:///{args.output}/optuna.db",
        load_if_exists=True,
        study_name=os.path.basename(args.output),
        direction="maximize",
    )

    # seed optimization with config params
    study.enqueue_trial(load_params_from_config(args))

    study.optimize(
        partial(objective, args=args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
