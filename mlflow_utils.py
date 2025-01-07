import mlflow
import mlflow.entities
import mlflow.entities.experiment


def create_experiment(name, artifact_location, tags):
    """
    Create an experiment with the given name, artifact location and tags.
    """
    try:
        experiment_id = mlflow.create_experiment(name=name,
                                 artifact_location=artifact_location,
                                 tags = tags)
        return experiment_id
    except Exception as e:
        print(f'The Experiment {name} already exists')
        experiment_id = mlflow.get_experiment_by_name(name).experiment_id
        return experiment_id


def get_experiment(name:str=None,experiment_id:str=None)->mlflow.entities.experiment:
    """
    Get experiment by name or experiment_id
    """
    if name is not None:
        experiment = mlflow.get_experiment_by_name(name)
    elif experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    else:
        raise ValueError('Either name or experiment_id must be provided')
    return experiment
