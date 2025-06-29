OPENAI_API_KEY=$OPENROUTER_API_KEY uv run run_optuna.py studies/sums_and_differences/trial0/checkpoints/checkpoint_250/best_program.py \
    examples/sums_and_differences/evaluator.py \
    --config examples/sums_and_differences/config.yaml \
    --output studies/sums_and_differences \
    --n_trials 1 \
    --n_jobs 1