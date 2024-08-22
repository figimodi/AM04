import subprocess
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, help='Configuration Path', required=True, dest='CONFIG')

    with open(args.CONFIG) as f:
        d = yaml.safe_load(f)
        config = Config(**d)

    # Train TSAI network
    harmoninaztion_training = [
        ('clear_old_generations.py'),
        ('generate_spattering_images.py', 
            '--min_dist', config.spattering.min_dist,
            '--max_points', config.spattering.max_points,
            '--max_spread', config.spattering.max_spread,
            '--darkest_gray', config.spattering.darkest_gray,
            '--lightest_gray', config.spattering.lightest_gray),
        ('generate_color_transferred_images.py',),
        ('train_harmonization.py', '--config', config.harmonization.config),
    ]


    for script in harmoninaztion_training:
        script_name = script[0]
        arguments = script[1:]
        command = ['python', script_name] + list(arguments)        
        subprocess.run(command)

    best_loss = 1e9
    best_model= None
    for file in os.listdir('log/train_harmonization/version_1'):
        if 'val_loss' in file:
            loss = int(file.split('.')[-2])
            if loss < best_loss:
                best_model= file

    best_model= os.path.join('log/train_harmonization/version_1/', best_model)

    # Generate synthetic images and pass them through pretrained TSAI network
    harmoninaztion_synthetic = [
        ('generate_synthetic_images.py', '--tot_samples', config.synthetic.n_samples),
        ('harmonize_synthetic_images.py',
            '--config', config.synthetic.config,
            '--only_test',
            '--pretrained', best_model),
    ]

    for script in harmoninaztion_synthetic:
        script_name = script[0]
        arguments = script[1:]
        command = ['python', script_name] + list(arguments)        
        subprocess.run(command)

    # Train classifier on the augmented dataset
    classification = [
        ('train_classifier.py', '--config', config.classification.config),
        ('tensorboard', 
            '--logdir', config.classification.log_dir,
            '--port', config.classification.port),
    ]

    for script in classification:
        script_name = script[0]
        arguments = script[1:]
        command = ['python', script_name] + list(arguments)        
        subprocess.run(command)
        