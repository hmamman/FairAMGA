import argparse
import time
import joblib

from utils.ml_classifiers import CLASSIFIERS
from utils.helpers import get_config_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--approach_name', type=str, default='fairsift', required=False, help='The name of fairness testing approach to run')
    parser.add_argument('--max_allowed_time', type=int, default=3600, help='Maximum time allowed for the experiment')
    parser.add_argument('--max_iteration', type=int, default=1, help='Maximum experiment iterations')
    return parser.parse_args()

def run_approach(approach_name, config, model, sensitive_param, max_time=3600):
    """Run the specified fairness testing approach."""
    if approach_name == 'fairamga':
        from tutorial.fairamga import FairAMGA
        FairAMGA(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'aequitas':
        from baseline.aequitas.aequitas import Aequitas
        Aequitas(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'expga':
        from baseline.expga.expga import ExpGA
        ExpGA(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'sbft':
        from baseline.sbft.sbft import SBFT
        SBFT(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)


args = parse_arguments()
approaches = ['fairamga', 'aequitas', 'expga', 'sbft']

if args.approach_name not in approaches:
    raise ValueError(f"Invalid sensitive name: {args.approach_name}. Available options are: {approaches}")

for _ in range(args.max_iteration):
    for config in get_config_dict().values():
        classifier_names = list(CLASSIFIERS.keys()) + ['dnn']
        for classifier_name in classifier_names:
            for sensitive_param in config.sens_name:
                print(f'Approach name: {args.approach_name}')
                print(f'Dataset: {config.dataset_name}')
                print(f'Classifier: {classifier_name}')
                print(f'Sensitive name: {config.sens_name[sensitive_param]}')

                if classifier_name == 'dnn':
                    import tensorflow as tf
                    classifier_path = f'models/{config.dataset_name}/dnn_slfc.keras'
                    model = tf.keras.models.load_model(classifier_path)
                else:
                    classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
                    model = joblib.load(classifier_path)

                run_approach(
                    approach_name=args.approach_name,
                    config=config,
                    model=model,
                    sensitive_param=sensitive_param,
                    max_time=args.max_allowed_time
                )
