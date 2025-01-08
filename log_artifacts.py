import mlflow
from ModelTraining.mlflow_utils import get_experiment

if __name__ == '__main__':

    experiment = mlflow.get_experiment_by_name('FakeNewsClassification')

    with mlflow.start_run(run_name = 'LSTM',experiment_id = experiment.experiment_id) as run:

        

        print('run_id: {}'.format(run.info.run_id))
        print('run_name: {}'.format(run.info.run_name))
        print('experiment_id: {}'.format(run.info.experiment_id))
        print('artifact_uri: {}'.format(run.info.artifact_uri))
        print('Start_time: {}'.format(run.info.start_time))
        print('End_time: {}'.format(run.info.end_time))