class Best_params():
    def __init__(self):
        self.basic_tuning_lists = {
            'learning_rate': [0.01, 0.005, 0.001, 0.0005, 0.0001],
            'beta': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
            'tau': [1, 0.5, 0.1, 0.05, 0.01]
        }
        
        self.reduced_tuning_lists = {
            'ml-25m': {
                'LightGCN': {
                    'learning_rate': [0.001],
                    'beta': [0.1, 0.5],
                    'tau': [0.05],
                },
                'SimpleX': {
                    'learning_rate': [0.0005, 0.005, 0.01],
                    'beta': [0.001, 0.005, 0.05, 0.1],
                    'tau': [0.1, 0.5, 1],
                },
                'NeuMF': {
                    'learning_rate': [0.005, 0.01],
                    'beta': [1],
                    'tau': [0.5, 1],
                },
                'FM': {
                    'learning_rate': [0.01],
                    'beta': [0.001, 0.005, 0.01, 0.05],
                    'tau': [0.01, 0.05, 0.1, 1],
                },
                'XSimGCL': {
                    'learning_rate': [0.0001],
                    'beta': [0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    'tau': [0.1],
                },
                'LightGCL': {
                    'learning_rate': [0.001],
                    'beta': [0.005, 0.01, 0.1, 0.5],
                    'tau': [0.05],
                }
            },
            'amazon-CDs_and_Vinyl': {
                'LightGCN': {
                    'learning_rate': [0.005, 0.001, 0.0001],
                    'beta': [0.001, 0.1, 1, 0.05],
                    'tau': [0.05, 0.1, 0.5],
                },
                'SimpleX': {
                    'learning_rate': [0.0001, 0.0005],
                    'beta': [0.05, 0.1],
                    'tau': [0.1, 0.5],
                },
                'NeuMF': {
                    'learning_rate': [0.005, 0.01],
                    'beta': [0.001],
                    'tau': [0.01, 0.5, 1],
                },
                'FM': {
                    'learning_rate': [0.0001, 0.0005, 0.01],
                    'beta': [0.001, 0.005],
                    'tau': [0.05, 0.1, 1],
                },
                'XSimGCL': {
                    'learning_rate': [0.005, 0.01],
                    'beta': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
                    'tau': [0.05],
                },
                'LightGCL': {
                    'learning_rate': [0.0001],
                    'beta': [0.001, 0.005, 0.1, 0.5, 1],
                    'tau': [1],
                }
            }
        }
        
        self.basic_params = {
            'ml-25m': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.0005, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2021': {'learning_rate': 0.0005, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2022': {'learning_rate': 0.0001, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2023': {'learning_rate': 0.0005, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2024': {'learning_rate': 0.0001, 'n_layers': 2, 'reg_weight': 1e-05}
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.01, 'gamma': 0.3, 'history_len': 30},
                    '2021': {'learning_rate': 0.01, 'gamma': 0.3, 'history_len': 30},
                    '2022': {'learning_rate': 0.0005, 'gamma': 0.3, 'history_len': 50},
                    '2023': {'learning_rate': 0.01, 'gamma': 0.3, 'history_len': 20},
                    '2024': {'learning_rate': 0.0005, 'gamma': 0.3, 'history_len': 50}
                },
                'NeuMF': {
                    '2020': {'learning_rate': 0.0005, 'dropout_prob': 0.0, 'mlp_hidden_size': '[32,16,8]'},
                    '2021': {'learning_rate': 0.0005, 'dropout_prob': 0.0, 'mlp_hidden_size': '[32,16,8]'},
                    '2022': {'learning_rate': 0.0005, 'dropout_prob': 0.0, 'mlp_hidden_size': '[64,32,16]'},
                    '2023': {'learning_rate': 0.0005, 'dropout_prob': 0.0, 'mlp_hidden_size': '[64,32,16]'},
                    '2024': {'learning_rate': 0.005, 'dropout_prob': 0.0, 'mlp_hidden_size': '[32,16,8]'}
                },
                'FM': {
                    '2020': {'learning_rate': 0.01},
                    '2021': {'learning_rate': 0.01},
                    '2022': {'learning_rate': 0.01},
                    '2023': {'learning_rate': 0.01},
                    '2024': {'learning_rate': 0.01}
                },
                # 'XSimGCL': {
                #     '2020': {'reg_weight': 0.0001, 'lambda': 0.05, 'eps': 1, 'temperature': 0.2, 'layer_cl': 2},
                #     '2021': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 1, 'temperature': 0.2, 'layer_cl': 2},
                #     '2022': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 1, 'temperature': 0.2, 'layer_cl': 2},
                #     '2023': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 1, 'temperature': 0.2, 'layer_cl': 2},
                #     '2024': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 1, 'temperature': 0.2, 'layer_cl': 1}
                # },  
                'XSimGCL': {
                    '2020': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.10, 'temperature': 0.2, 'layer_cl': 1},
                    '2021': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.10, 'temperature': 0.2, 'layer_cl': 1},
                    '2022': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.10, 'temperature': 0.2, 'layer_cl': 1},
                    '2023': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.10, 'temperature': 0.2, 'layer_cl': 1},
                    '2024': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.10, 'temperature': 0.2, 'layer_cl': 1}
                },
                'LightGCL': {
                    '2020': {'dropout': 0, 'temp': 0.3, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2021': {'dropout': 0, 'temp': 3, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2022': {'dropout': 0, 'temp': 10, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2023': {'dropout': 0, 'temp': 3, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2024': {'dropout': 0.25, 'temp': 0.3, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5}
                },
                'BPR': {
                    '2020': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2021': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2022': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2023': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2024': {'learning_rate': 0.005, 'reg_weight': 0.001}
                }
            },
            'amazon-CDs_and_Vinyl': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.005, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2021': {'learning_rate': 0.005, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2022': {'learning_rate': 0.005, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2023': {'learning_rate': 0.005, 'n_layers': 2, 'reg_weight': 1e-05},
                    '2024': {'learning_rate': 0.005, 'n_layers': 2, 'reg_weight': 1e-05}
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.0005, 'gamma': 0.3, 'history_len': 0},
                    '2021': {'learning_rate': 0.0001, 'gamma': 0.3, 'history_len': 0},
                    '2022': {'learning_rate': 0.0005, 'gamma': 0.3, 'history_len': 0},
                    '2023': {'learning_rate': 0.0001, 'gamma': 0.3, 'history_len': 0},
                    '2024': {'learning_rate': 0.0001, 'gamma': 0.3, 'history_len': 0}
                },
                'NeuMF': {
                    '2020': {'learning_rate': 0.0005, 'dropout_prob': 0.0, 'mlp_hidden_size': '[32,16,8]'},
                    '2021': {'learning_rate': 0.001, 'dropout_prob': 0.0, 'mlp_hidden_size': '[64,32,16]'},
                    '2022': {'learning_rate': 0.001, 'dropout_prob': 0.0, 'mlp_hidden_size': '[64,32,16]'},
                    '2023': {'learning_rate': 0.001, 'dropout_prob': 0.0, 'mlp_hidden_size': '[64,32,16]'},
                    '2024': {'learning_rate': 0.001, 'dropout_prob': 0.0, 'mlp_hidden_size': '[64,32,16]'}
                },
                'FM': {
                    '2020': {'learning_rate': 0.0005},
                    '2021': {'learning_rate': 0.0001},
                    '2022': {'learning_rate': 0.0001},
                    '2023': {'learning_rate': 0.0001},
                    '2024': {'learning_rate': 0.0001}
                },
                'XSimGCL': {
                    '2020': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.05, 'temperature': 0.2, 'layer_cl': 2},
                    '2021': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.05, 'temperature': 0.2, 'layer_cl': 2},
                    '2022': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.05, 'temperature': 0.2, 'layer_cl': 2},
                    '2023': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.05, 'temperature': 0.2, 'layer_cl': 2},
                    '2024': {'reg_weight': 0.0001, 'lambda': 0.01, 'eps': 0.05, 'temperature': 0.2, 'layer_cl': 2}
                },
                'LightGCL': {
                    '2020': {'dropout': 0, 'temp': 10, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2021': {'dropout': 0, 'temp': 0.3, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2022': {'dropout': 0, 'temp': 1, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2023': {'dropout': 0, 'temp': 0.5, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5},
                    '2024': {'dropout': 0, 'temp': 0.5, 'lambda1': 1e-05, 'lambda2': 1e-05, 'q': 5}
                },
                'BPR': {
                    '2020': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2021': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2022': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2023': {'learning_rate': 0.005, 'reg_weight': 0.001},
                    '2024': {'learning_rate': 0.005, 'reg_weight': 0.001}
                }
            },
        }
        self.cos_noadd_params_1 = {
            'ml-25m': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 1},
                    '2021': {'learning_rate': 0.01, 'beta': 0.1, 'tau': 1},
                    '2022': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 0.5},
                    '2023': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.5},
                    '2024': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 0.5},
                },
                'FM': {
                    '2020': {'learning_rate': 0.01, 'beta': 0.005, 'tau': 0.1},
                    '2021': {'learning_rate': 0.01, 'beta': 0.01, 'tau': 1},
                    '2022': {'learning_rate': 0.01, 'beta': 0.01, 'tau': 1},
                    '2023': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.05},
                    '2024': {'learning_rate': 0.01, 'beta': 0.005, 'tau': 0.05}
                },
                'XSimGCL': {
                    '2020': {'learning_rate': 0.0001, 'beta': 1, 'tau': 0.1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.005, 'tau': 0.1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 0.1},
                    '2024': {'learning_rate': 0.0001, 'beta': 1, 'tau': 0.1},
                },
                'LightGCL': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.01, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.005, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                }
            },
            'amazon-CDs_and_Vinyl': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.005, 'beta': 0.001, 'tau': 0.5},
                    '2021': {'learning_rate': 0.005, 'beta': 1, 'tau': 0.05},
                    '2022': {'learning_rate': 0.005, 'beta': 1, 'tau': 0.05},
                    '2023': {'learning_rate': 0.005, 'beta': 1, 'tau': 0.05},
                    '2024': {'learning_rate': 0.005, 'beta': 1, 'tau': 0.05},
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.0005, 'beta': 0.1, 'tau': 0.1},
                    '2021': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                    '2022': {'learning_rate': 0.0005, 'beta': 0.1, 'tau': 0.1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.1},
                    '2024': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                },
                'FM': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2023': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                },
                'XSimGCL': {
                    '2020': {'learning_rate': 0.005, 'beta': 0.01, 'tau': 0.05},
                    '2021': {'learning_rate': 0.005, 'beta': 0.001, 'tau': 0.05},
                    '2022': {'learning_rate': 0.005, 'beta': 0.1, 'tau': 0.05},
                    '2023': {'learning_rate': 0.005, 'beta': 0.005, 'tau': 0.05},
                    '2024': {'learning_rate': 0.005, 'beta': 0.01, 'tau': 0.05},
                },
                'LightGCL': {
                    '2020': {'learning_rate': 0.0001, 'beta': 1, 'tau': 1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.005, 'tau': 1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 1}
                }
            },
        }
        self.cos_noadd_params_3 = {
            'ml-25m': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.005, 'beta': 0.1, 'tau': 1},
                    '2021': {'learning_rate': 0.005, 'beta': 0.1, 'tau': 0.5},
                    '2022': {'learning_rate': 0.01, 'beta': 0.005, 'tau': 0.5},
                    '2023': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.5},
                    '2024': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 0.1}
                },
                'FM': {
                    '2020': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.05},
                    '2021': {'learning_rate': 0.01, 'beta': 0.005, 'tau': 0.1},
                    '2022': {'learning_rate': 0.01, 'beta': 0.01, 'tau': 0.1},
                    '2023': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 0.05},
                    '2024': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.05},
                },
                'XSimGCL': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.1, 'tau': 0.1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.1, 'tau': 0.1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 0.1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.005, 'tau': 0.1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.1, 'tau': 0.1},
                },
                'LightGCL': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.005, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.005, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.005, 'tau': 0.05},
                }
            },
            'amazon-CDs_and_Vinyl': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.005, 'beta': 0.001, 'tau': 0.05},
                    '2021': {'learning_rate': 0.005, 'beta': 0.001, 'tau': 0.05},
                    '2022': {'learning_rate': 0.005, 'beta': 1, 'tau': 0.05},
                    '2023': {'learning_rate': 0.005, 'beta': 0.001, 'tau': 0.1},
                    '2024': {'learning_rate': 0.005, 'beta': 1, 'tau': 0.05},
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.1, 'tau': 0.1},
                    '2022': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                    '2023': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.5},
                },
                'FM': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2024': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 1},
                },
                'XSimGCL': {
                    '2020': {'learning_rate': 0.005, 'beta': 0.5, 'tau': 0.05},
                    '2021': {'learning_rate': 0.005, 'beta': 0.1, 'tau': 0.05},
                    '2022': {'learning_rate': 0.005, 'beta': 0.5, 'tau': 0.05},
                    '2023': {'learning_rate': 0.005, 'beta': 0.5, 'tau': 0.05},
                    '2024': {'learning_rate': 0.005, 'beta': 0.5, 'tau': 0.05}
                },
                'LightGCL': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.1, 'tau': 1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2023': {'learning_rate': 0.0001, 'beta': 1, 'tau': 1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                }
            },
        }
        self.cos_noadd_params_5 = {
            'ml-25m': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.5},
                    '2021': {'learning_rate': 0.005, 'beta': 0.1, 'tau': 1},
                    '2022': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 0.5},
                    '2023': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 1},
                    '2024': {'learning_rate': 0.005, 'beta': 0.1, 'tau': 1},
                },
                'FM': {
                    '2020': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.05},
                    '2021': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 0.1},
                    '2022': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.05},
                    '2023': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 0.05},
                    '2024': {'learning_rate': 0.01, 'beta': 0.01, 'tau': 0.1},
                },
                'XSimGCL': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.01, 'tau': 0.1},
                    '2021': {'learning_rate': 0.0001, 'beta': 1, 'tau': 0.1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.005, 'tau': 0.1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 0.1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.1},
                },
                'LightGCL': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.01, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.005, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                }
            },
            'amazon-CDs_and_Vinyl': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.5},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.5},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.5},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.5},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.5},
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.05, 'tau': 0.1},
                    '2022': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                    '2023': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                    '2024': {'learning_rate': 0.0005, 'beta': 0.05, 'tau': 0.1},
                },
                'FM': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2022': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2024': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 1},
                },
                'XSimGCL': {
                    '2020': {'learning_rate': 0.005, 'beta': 0.1, 'tau': 0.05},
                    '2021': {'learning_rate': 0.005, 'beta': 0.01, 'tau': 0.05},
                    '2022': {'learning_rate': 0.005, 'beta': 0.01, 'tau': 0.05},
                    '2023': {'learning_rate': 0.005, 'beta': 0.05, 'tau': 0.05},
                    '2024': {'learning_rate': 0.005, 'beta': 0.01, 'tau': 0.05}
                },
                'LightGCL': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.001, 'tau': 1},
                    '2021': {'learning_rate': 0.0001, 'beta': 1, 'tau': 1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.1, 'tau': 1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.005, 'tau': 1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 1}
                }
            },
        }
        self.cos_noadd_params_10 = {
            'ml-25m': {
                'LightGCN': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.1, 'tau': 0.05},
                },
                'SimpleX': {
                    '2020': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.5},
                    '2021': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 1},
                    '2022': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 0.5},
                    '2023': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 1},
                    '2024': {'learning_rate': 0.0005, 'beta': 0.001, 'tau': 0.1},
                },
                'FM': {
                    '2020': {'learning_rate': 0.01, 'beta': 0.005, 'tau': 0.1},
                    '2021': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 1},
                    '2022': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 0.05},
                    '2023': {'learning_rate': 0.01, 'beta': 0.001, 'tau': 0.05},
                    '2024': {'learning_rate': 0.01, 'beta': 0.05, 'tau': 0.05},
                },
                'XSimGCL': {
                    '2020': {'learning_rate': 0.0001, 'beta': 0.01, 'tau': 0.1},
                    '2021': {'learning_rate': 0.0001, 'beta': 0.005, 'tau': 0.1},
                    '2022': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 0.1},
                    '2023': {'learning_rate': 0.0001, 'beta': 0.5, 'tau': 0.1},
                    '2024': {'learning_rate': 0.0001, 'beta': 0.005, 'tau': 0.1},
                },
                'LightGCL': {
                    '2020': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2021': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2022': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                    '2023': {'learning_rate': 0.001, 'beta': 0.005, 'tau': 0.05},
                    '2024': {'learning_rate': 0.001, 'beta': 0.5, 'tau': 0.05},
                }
            },
        }