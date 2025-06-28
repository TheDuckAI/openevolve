OPENAI_API_KEY=$OPENROUTER_API_KEY uv run run_optuna.py examples/sphere_packing/initial_program_2.py \
    examples/sphere_packing/evaluator.py \
    --config examples/sphere_packing/config.yaml \
    --output studies/sphere_packing \
    --n_trials 20 \
    --n_jobs 5