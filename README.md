# MADRL for decentralized proactive transshipment
The implementation in this repositorory is used in the paper "Multi-Agent Deep Reinforcement Learning for Decentralized Proactive Transshipment"（{[placeholder](https://openai.com)}）
The repository is based on (<https://github.com/marlbenchmark/on-policy>, <https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management>), and we introduce some techniques tailored for inventory management problem.
The dataset we used in the paper is from <https://www.coap.online/competitions/1>

# How to Run
If you want to train a model under default setting, you can run the following command:
```
python train_env.py --yaml_path ./setting_yaml/multi_discrete/default.yaml
```
If you want to train a model under a real-world dataset we used in the paper, you can run the following command:
```
python train_env.py --yaml_path ./setting_yaml/multi_discrete/SKU.yaml
``` 
For further adjustments regarding model training or environment parameters, you can refer to the ".yaml" files in the "setting_yaml" folder to make modifications. Additionally, the "config.py" file contains more adjustable parameters.