num_classes = 10
settings = {
    'FashionMNIST': {
        'FGSM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'FGM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.8},
        },
        'StepLL': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.035},
        },
        'IFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.05},
        },
        'MIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.05},
        },
        'NIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.05},
        },
        'PGD': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.035},
        },
        'IterLL': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.012},
        },
        'DeepFool': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {},
        },
        'LBFGS': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'c': 0.05},
        },
        'OnePixel': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {'popsize': 400},
        },
        'CW': {
            'max_iter': 100, 
            'attack_kwargs': {'c': 0.05, 'lr': 0.01}
        }
    },
    
    'CIFAR10': {
        'FGSM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'FGM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 1},
        },
        'StepLL': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'MIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'NIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'PGD': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IterLL': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.012},
        },
        'DeepFool': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {},
        },
        'LBFGS': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'c': 0.06},
        },
        'OnePixel': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {'popsize': 400},
        },
        'CW': {
            'max_iter': 100, 
            'attack_kwargs': {'c': 3, 'lr': 0.02}
        },
    },
    
    'Imagenette': {
        'FGSM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'FGM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 5},
        },
        'StepLL': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'MIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'NIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'PGD': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IterLL': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.012},
        },
        'DeepFool': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {},
        },
        'LBFGS': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'c': 0.3},
        },
        'OnePixel': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {'popsize': 400},
        }
    },
}