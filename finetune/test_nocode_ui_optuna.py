def test1():
    import optuna
    from optuna_dashboard import run_dashboard

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        # Train model with these parameters
        return validation_map

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Launch dashboard
    run_dashboard(study)  # Opens at http://localhost:8080
def test2():
    import optuna
    import optunahub


    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_float("y", -5, 5)
        return x**2 + y**2


    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(sampler=module.AutoSampler())
    study.optimize(objective, n_trials=10)

    print(study.best_trial.value, study.best_trial.params)

def test3():
    import optuna

    def objective(trial):
        x1 = trial.suggest_float("x1", -100, 100)
        x2 = trial.suggest_float("x2", -100, 100)
        return x1**2 + 0.01 * x2**2


    study = optuna.create_study(storage="sqlite:///db.sqlite3")  # Create a new study with database.
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    test3()