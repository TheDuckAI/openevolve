OPENAI_API_KEY=$OPENROUTER_API_KEY uv run run_optuna.py examples/first_autocorrelation/initial_program.py \
    examples/first_autocorrelation/evaluator.py \
    --config examples/first_autocorrelation/config_1.yaml \
    --output studies/first_autocorrelation \
    --n_trials 10 \
    --n_jobs 10