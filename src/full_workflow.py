import subprocess
import yaml
import os
from pathlib import Path
from argparse import ArgumentParser
from pydantic import BaseModel
from typing import Optional

    
class SpatteringConfig(BaseModel):
    min_dist: int
    max_points: int
    max_spread: int
    darkest_gray: int
    lightest_gray: int

class HarmonizationConfig(BaseModel):
    config: Path

class ClassificationConfig(BaseModel):
    config: Path
    classify: Optional[bool] = True

class SyntheticConfig(BaseModel):
    n_samples: int
    config: Path
    pretrained: Optional[Path] = None
    generate: Optional[bool] = True

class TensorboardConfig(BaseModel):
    log_dir: Path
    port: int

class Config(BaseModel):
    spattering: SpatteringConfig
    harmonization: HarmonizationConfig
    synthetic: SyntheticConfig
    classification: ClassificationConfig
    tensorboard: TensorboardConfig

def launch_scripts(venv, scripts):
    for script in scripts:
        if isinstance(script, list):
            script_name = script[0]
            arguments = script[1:]
            command = [venv, script_name] + list(arguments)        
        else:
            command = [venv, script]

        subprocess.run(command)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, help='Configuration Path', required=True, dest='CONFIG')
    args = parser.parse_args()

    # Set path to the python executable, leave 'python' if you're not using any venv
    venv = '../.venv/Scripts/python'

    with open(args.CONFIG) as f:
        d = yaml.safe_load(f)
        config = Config(**d)
   
    if config.synthetic.pretrained:
        # Load already pretrained TSAI network
        best_model = config.synthetic.pretrained
    else:
        # Train TSAI network
        harmoninaztion_training = [
            "clear_old_generations.py",
            ['generate_spattering_images.py', 
                '--min_dist', str(config.spattering.min_dist),
                '--max_points', str(config.spattering.max_points),
                '--max_spread', str(config.spattering.max_spread),
                '--darkest_gray', str(config.spattering.darkest_gray),
                '--lightest_gray', str(config.spattering.lightest_gray)],
            'generate_color_transferred_images.py',
            ['train_harmonization.py', '--config', str(config.harmonization.config)],
        ]

        launch_scripts(venv, harmoninaztion_training) 

        best_loss = 1e9
        best_model= None
        for file in os.listdir('./log/train_harmonization/version_1'):
            if 'val_loss' in file:
                loss = int(file.split('.')[-2])
                if loss < best_loss:
                    best_model= file

        best_model= os.path.join('log/train_harmonization/version_1/', best_model)

    if config.synthetic.generate:
        # Generate synthetic images and pass them through pretrained TSAI network
        harmoninaztion_synthetic = [
            ['generate_synthetic_images.py', '--tot_samples', str(config.synthetic.n_samples)],
            ['harmonize_synthetic_images.py',
                '--config', str(config.synthetic.config),
                '--only_test',
                '--pretrained', str(best_model)],
        ]

        launch_scripts(venv, harmoninaztion_synthetic)

    if config.classification.classify:
        # Train classifier on the augmented dataset
        classification = [
            ['train_classifier.py', '--config', str(config.classification.config)],
            ['tensorboard', 
                '--logdir', str(config.classification.log_dir),
                '--port', str(config.classification.port)],
        ]

        launch_scripts(venv, classification)
            