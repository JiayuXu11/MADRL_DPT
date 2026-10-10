#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python train.py --load_config configs/backlog_additional/default.json --exp_name backlog_additional_default --seed 1
python train.py --load_config configs/backlog_additional/default.json --exp_name backlog_additional_default --seed 2
python train.py --load_config configs/backlog_additional/default.json --exp_name backlog_additional_default --seed 3
python train.py --load_config configs/backlog_additional/default.json --exp_name backlog_additional_default --seed 4
python train.py --load_config configs/backlog_additional/default.json --exp_name backlog_additional_default --seed 5

python train.py --load_config configs/backlog_additional/h02.json --exp_name backlog_additional_h02 --seed 1
python train.py --load_config configs/backlog_additional/h02.json --exp_name backlog_additional_h02 --seed 2
python train.py --load_config configs/backlog_additional/h02.json --exp_name backlog_additional_h02 --seed 3
python train.py --load_config configs/backlog_additional/h02.json --exp_name backlog_additional_h02 --seed 4
python train.py --load_config configs/backlog_additional/h02.json --exp_name backlog_additional_h02 --seed 5

python train.py --load_config configs/backlog_additional/h03.json --exp_name backlog_additional_h03 --seed 1
python train.py --load_config configs/backlog_additional/h03.json --exp_name backlog_additional_h03 --seed 2
python train.py --load_config configs/backlog_additional/h03.json --exp_name backlog_additional_h03 --seed 3
python train.py --load_config configs/backlog_additional/h03.json --exp_name backlog_additional_h03 --seed 4
python train.py --load_config configs/backlog_additional/h03.json --exp_name backlog_additional_h03 --seed 5

python train.py --load_config configs/backlog_additional/h04.json --exp_name backlog_additional_h04 --seed 1
python train.py --load_config configs/backlog_additional/h04.json --exp_name backlog_additional_h04 --seed 2
python train.py --load_config configs/backlog_additional/h04.json --exp_name backlog_additional_h04 --seed 3
python train.py --load_config configs/backlog_additional/h04.json --exp_name backlog_additional_h04 --seed 4
python train.py --load_config configs/backlog_additional/h04.json --exp_name backlog_additional_h04 --seed 5

python train.py --load_config configs/backlog_additional/h05.json --exp_name backlog_additional_h05 --seed 1
python train.py --load_config configs/backlog_additional/h05.json --exp_name backlog_additional_h05 --seed 2
python train.py --load_config configs/backlog_additional/h05.json --exp_name backlog_additional_h05 --seed 3
python train.py --load_config configs/backlog_additional/h05.json --exp_name backlog_additional_h05 --seed 4
python train.py --load_config configs/backlog_additional/h05.json --exp_name backlog_additional_h05 --seed 5

python train.py --load_config configs/backlog_additional/b04.json --exp_name backlog_additional_b04 --seed 1
python train.py --load_config configs/backlog_additional/b04.json --exp_name backlog_additional_b04 --seed 2
python train.py --load_config configs/backlog_additional/b04.json --exp_name backlog_additional_b04 --seed 3
python train.py --load_config configs/backlog_additional/b04.json --exp_name backlog_additional_b04 --seed 4
python train.py --load_config configs/backlog_additional/b04.json --exp_name backlog_additional_b04 --seed 5

python train.py --load_config configs/backlog_additional/b06.json --exp_name backlog_additional_b06 --seed 1
python train.py --load_config configs/backlog_additional/b06.json --exp_name backlog_additional_b06 --seed 2
python train.py --load_config configs/backlog_additional/b06.json --exp_name backlog_additional_b06 --seed 3
python train.py --load_config configs/backlog_additional/b06.json --exp_name backlog_additional_b06 --seed 4
python train.py --load_config configs/backlog_additional/b06.json --exp_name backlog_additional_b06 --seed 5

python train.py --load_config configs/backlog_additional/b08.json --exp_name backlog_additional_b08 --seed 1
python train.py --load_config configs/backlog_additional/b08.json --exp_name backlog_additional_b08 --seed 2
python train.py --load_config configs/backlog_additional/b08.json --exp_name backlog_additional_b08 --seed 3
python train.py --load_config configs/backlog_additional/b08.json --exp_name backlog_additional_b08 --seed 4
python train.py --load_config configs/backlog_additional/b08.json --exp_name backlog_additional_b08 --seed 5

python train.py --load_config configs/backlog_additional/scpu0.json --exp_name backlog_additional_scpu0 --seed 1
python train.py --load_config configs/backlog_additional/scpu0.json --exp_name backlog_additional_scpu0 --seed 2
python train.py --load_config configs/backlog_additional/scpu0.json --exp_name backlog_additional_scpu0 --seed 3
python train.py --load_config configs/backlog_additional/scpu0.json --exp_name backlog_additional_scpu0 --seed 4
python train.py --load_config configs/backlog_additional/scpu0.json --exp_name backlog_additional_scpu0 --seed 5

python train.py --load_config configs/backlog_additional/scpu0001.json --exp_name backlog_additional_scpu0001 --seed 1
python train.py --load_config configs/backlog_additional/scpu0001.json --exp_name backlog_additional_scpu0001 --seed 2
python train.py --load_config configs/backlog_additional/scpu0001.json --exp_name backlog_additional_scpu0001 --seed 3
python train.py --load_config configs/backlog_additional/scpu0001.json --exp_name backlog_additional_scpu0001 --seed 4
python train.py --load_config configs/backlog_additional/scpu0001.json --exp_name backlog_additional_scpu0001 --seed 5

python train.py --load_config configs/backlog_additional/scpu00015.json --exp_name backlog_additional_scpu00015 --seed 1
python train.py --load_config configs/backlog_additional/scpu00015.json --exp_name backlog_additional_scpu00015 --seed 2
python train.py --load_config configs/backlog_additional/scpu00015.json --exp_name backlog_additional_scpu00015 --seed 3
python train.py --load_config configs/backlog_additional/scpu00015.json --exp_name backlog_additional_scpu00015 --seed 4
python train.py --load_config configs/backlog_additional/scpu00015.json --exp_name backlog_additional_scpu00015 --seed 5

python train.py --load_config configs/backlog_additional/scpu0002.json --exp_name backlog_additional_scpu0002 --seed 1
python train.py --load_config configs/backlog_additional/scpu0002.json --exp_name backlog_additional_scpu0002 --seed 2
python train.py --load_config configs/backlog_additional/scpu0002.json --exp_name backlog_additional_scpu0002 --seed 3
python train.py --load_config configs/backlog_additional/scpu0002.json --exp_name backlog_additional_scpu0002 --seed 4
python train.py --load_config configs/backlog_additional/scpu0002.json --exp_name backlog_additional_scpu0002 --seed 5

python train.py --load_config configs/backlog_additional/agents5.json --exp_name backlog_additional_agents5 --seed 1
python train.py --load_config configs/backlog_additional/agents5.json --exp_name backlog_additional_agents5 --seed 2
python train.py --load_config configs/backlog_additional/agents5.json --exp_name backlog_additional_agents5 --seed 3
python train.py --load_config configs/backlog_additional/agents5.json --exp_name backlog_additional_agents5 --seed 4
python train.py --load_config configs/backlog_additional/agents5.json --exp_name backlog_additional_agents5 --seed 5

python train.py --load_config configs/backlog_additional/agents7.json --exp_name backlog_additional_agents7 --seed 1
python train.py --load_config configs/backlog_additional/agents7.json --exp_name backlog_additional_agents7 --seed 2
python train.py --load_config configs/backlog_additional/agents7.json --exp_name backlog_additional_agents7 --seed 3
python train.py --load_config configs/backlog_additional/agents7.json --exp_name backlog_additional_agents7 --seed 4
python train.py --load_config configs/backlog_additional/agents7.json --exp_name backlog_additional_agents7 --seed 5

python train.py --load_config configs/backlog_additional/agents9.json --exp_name backlog_additional_agents9 --seed 1
python train.py --load_config configs/backlog_additional/agents9.json --exp_name backlog_additional_agents9 --seed 2
python train.py --load_config configs/backlog_additional/agents9.json --exp_name backlog_additional_agents9 --seed 3
python train.py --load_config configs/backlog_additional/agents9.json --exp_name backlog_additional_agents9 --seed 4
python train.py --load_config configs/backlog_additional/agents9.json --exp_name backlog_additional_agents9 --seed 5

python train.py --load_config configs/backlog_additional/moq0.json --exp_name backlog_additional_moq0 --seed 1
python train.py --load_config configs/backlog_additional/moq0.json --exp_name backlog_additional_moq0 --seed 2
python train.py --load_config configs/backlog_additional/moq0.json --exp_name backlog_additional_moq0 --seed 3
python train.py --load_config configs/backlog_additional/moq0.json --exp_name backlog_additional_moq0 --seed 4
python train.py --load_config configs/backlog_additional/moq0.json --exp_name backlog_additional_moq0 --seed 5

python train.py --load_config configs/backlog_additional/moq40.json --exp_name backlog_additional_moq40 --seed 1
python train.py --load_config configs/backlog_additional/moq40.json --exp_name backlog_additional_moq40 --seed 2
python train.py --load_config configs/backlog_additional/moq40.json --exp_name backlog_additional_moq40 --seed 3
python train.py --load_config configs/backlog_additional/moq40.json --exp_name backlog_additional_moq40 --seed 4
python train.py --load_config configs/backlog_additional/moq40.json --exp_name backlog_additional_moq40 --seed 5

python train.py --load_config configs/backlog_additional/moq60.json --exp_name backlog_additional_moq60 --seed 1
python train.py --load_config configs/backlog_additional/moq60.json --exp_name backlog_additional_moq60 --seed 2
python train.py --load_config configs/backlog_additional/moq60.json --exp_name backlog_additional_moq60 --seed 3
python train.py --load_config configs/backlog_additional/moq60.json --exp_name backlog_additional_moq60 --seed 4
python train.py --load_config configs/backlog_additional/moq60.json --exp_name backlog_additional_moq60 --seed 5

python train.py --load_config configs/backlog_additional/lt8.json --exp_name backlog_additional_lt8 --seed 1
python train.py --load_config configs/backlog_additional/lt8.json --exp_name backlog_additional_lt8 --seed 2
python train.py --load_config configs/backlog_additional/lt8.json --exp_name backlog_additional_lt8 --seed 3
python train.py --load_config configs/backlog_additional/lt8.json --exp_name backlog_additional_lt8 --seed 4
python train.py --load_config configs/backlog_additional/lt8.json --exp_name backlog_additional_lt8 --seed 5

python train.py --load_config configs/backlog_additional/lt11.json --exp_name backlog_additional_lt11 --seed 1
python train.py --load_config configs/backlog_additional/lt11.json --exp_name backlog_additional_lt11 --seed 2
python train.py --load_config configs/backlog_additional/lt11.json --exp_name backlog_additional_lt11 --seed 3
python train.py --load_config configs/backlog_additional/lt11.json --exp_name backlog_additional_lt11 --seed 4
python train.py --load_config configs/backlog_additional/lt11.json --exp_name backlog_additional_lt11 --seed 5

python train.py --load_config configs/backlog_additional/lt15.json --exp_name backlog_additional_lt15 --seed 1
python train.py --load_config configs/backlog_additional/lt15.json --exp_name backlog_additional_lt15 --seed 2
python train.py --load_config configs/backlog_additional/lt15.json --exp_name backlog_additional_lt15 --seed 3
python train.py --load_config configs/backlog_additional/lt15.json --exp_name backlog_additional_lt15 --seed 4
python train.py --load_config configs/backlog_additional/lt15.json --exp_name backlog_additional_lt15 --seed 5
