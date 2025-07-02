OPENAI_API_KEY=$OPENROUTER_API_KEY uv run run_optuna.py studies/sums_and_differences/trial17/checkpoints/checkpoint_30/best_program.py \
    examples/sums_and_differences/evaluator.py \
    --config examples/sums_and_differences/config_2.yaml \
    --output studies/sums_and_differences \
    --n_trials 10 \
    --n_jobs 10