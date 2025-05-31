def get_default_config():
    """
    Get default configuration for sentiment model training
    """
    config = {
        # Data settings
        "raw_data_path": "data/raw/training.1600000.processed.noemoticon.csv",
        "processed_data_path": "data/processed/processed_data.pt",
        "output_dir": "models/unstuck_model",
        "max_seq_length": 256,  # Increased from 128 to capture more context
        "max_samples": 100000,  # Set reasonable sample size for training
        
        # Model parameters
        "num_labels": 5,  # 5-class sentiment
        "hidden_dropout_prob": 0.1,
        "attention_dropout_prob": 0.1,
        "classifier_dropout": 0.2,
        "freeze_base_layers": True,
        "unfreeze_last_n_layers": 3,  # Increased to allow more fine-tuning
        
        # Training hyperparameters
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "epochs": 5,
        "learning_rate": 5e-05,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 3,
        "save_steps": 100,
        
        # Advanced training features
        "enable_advanced_class_balancing": True,  # Enable class balancing
        "class_balance_strategy": "synthetic",  # Options: 'synthetic', 'weighted', 'sample'
        "enable_data_augmentation": True,  # Enable data augmentation
        "augmentation_techniques": ["backtranslation", "synonym_replacement", "random_swap"],
        "augmentation_factor": 0.3,  # Augment 30% of examples
        "target_class_distribution": {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},  # Balanced
        
        # Optimizer settings
        "loss_function": "focal_loss",  # Better for imbalanced classes
        "optimizer": "adamw_8bit",
        "lr_scheduler": "linear_with_restart",
        "dynamic_lr_scaling": True,
        "loss_patience_threshold": 10,
        "min_loss_change_threshold": 1e-05,
        "lr_scaling_factor": 1.5,
        "track_loss_history": True,
        
        # Hardware settings
        "use_cuda": True,
        "use_amp": True,  # Mixed precision training
        "seed": None  # Will be set to random value if None
    }
    
    return config