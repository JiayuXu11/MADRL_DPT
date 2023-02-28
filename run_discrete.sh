python train_env.py --yaml_path "setting_yaml/discrete/discrete_no_info_sharing.yaml"
# relu
python train_env.py --yaml_path "setting_yaml/discrete/discrete_no_info_sharing.yaml" --experiment_name discrete_no_info_sharing_relu --use_ReLU True
# adavantage scaling
python train_env.py --yaml_path "setting_yaml/discrete/discrete_no_info_sharing.yaml" --experiment_name discrete_no_info_sharing_tanh_adv_scale --sample_mean_advantage False
# instant info sharing
python train_env.py --yaml_path "setting_yaml/discrete/discrete_no_info_sharing.yaml" --experiment_name discrete_no_info_sharing_tanh_instant --instant_info_sharing True
# ratio transship
python train_env.py --yaml_path "setting_yaml/discrete/discrete_no_info_sharing.yaml" --experiment_name discrete_no_info_sharing_tanh_ratio --ratio_transship True
# pure_returns
python train_env.py --yaml_path "setting_yaml/discrete/discrete_no_info_sharing.yaml" --experiment_name discrete_no_info_sharing_tanh_pure --critic_learning_pure_returns True
# pure_returns and scaling
python train_env.py --yaml_path "setting_yaml/discrete/discrete_no_info_sharing.yaml" --experiment_name discrete_no_info_sharing_tanh_pure_scale --critic_learning_pure_returns True --sample_mean_advantage False

