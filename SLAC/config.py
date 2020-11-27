Trainer_config = {
    'initial_collection_steps': 10000,
    'initial_learning_steps': 3,
    'num_sequences': 8,
    'eval_interval': 10000,
    'num_eval_episodes': 5
}

SLAC_config = {
    'gamma': 0.99,
    'batch_size_sac': 256,
    'batch_size_latent': 32,
    'buffer_size': 10 ** 5,
    'num_sequences': 8,
    'lr_sac': 3e-4,
    'lr_latent': 1e-4,
    'feature_dim': 256,
    'z1_dim': 32,
    'z2_dim': 256,
    'hidden_units': (256, 256),
    'tau': 5e-3
}
LOG_INTERVAL = 10  # frequency to write summary
