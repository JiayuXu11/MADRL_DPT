#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python train.py --load_config configs/backlog/default.json --exp_name backlog_default --seed 1
python train.py --load_config configs/backlog/default.json --exp_name backlog_default --seed 2
python train.py --load_config configs/backlog/default.json --exp_name backlog_default --seed 3
python train.py --load_config configs/backlog/default.json --exp_name backlog_default --seed 4
python train.py --load_config configs/backlog/default.json --exp_name backlog_default --seed 5

python train.py --load_config configs/backlog/agents12.json --exp_name backlog_agents12 --seed 1
python train.py --load_config configs/backlog/agents12.json --exp_name backlog_agents12 --seed 2
python train.py --load_config configs/backlog/agents12.json --exp_name backlog_agents12 --seed 3
python train.py --load_config configs/backlog/agents12.json --exp_name backlog_agents12 --seed 4
python train.py --load_config configs/backlog/agents12.json --exp_name backlog_agents12 --seed 5

python train.py --load_config configs/backlog/agents15.json --exp_name backlog_agents15 --seed 1
python train.py --load_config configs/backlog/agents15.json --exp_name backlog_agents15 --seed 2
python train.py --load_config configs/backlog/agents15.json --exp_name backlog_agents15 --seed 3
python train.py --load_config configs/backlog/agents15.json --exp_name backlog_agents15 --seed 4
python train.py --load_config configs/backlog/agents15.json --exp_name backlog_agents15 --seed 5

python train.py --load_config configs/backlog/agents2.json --exp_name backlog_agents2 --seed 1
python train.py --load_config configs/backlog/agents2.json --exp_name backlog_agents2 --seed 2
python train.py --load_config configs/backlog/agents2.json --exp_name backlog_agents2 --seed 3
python train.py --load_config configs/backlog/agents2.json --exp_name backlog_agents2 --seed 4
python train.py --load_config configs/backlog/agents2.json --exp_name backlog_agents2 --seed 5

python train.py --load_config configs/backlog/agents20.json --exp_name backlog_agents20 --seed 1
python train.py --load_config configs/backlog/agents20.json --exp_name backlog_agents20 --seed 2
python train.py --load_config configs/backlog/agents20.json --exp_name backlog_agents20 --seed 3
python train.py --load_config configs/backlog/agents20.json --exp_name backlog_agents20 --seed 4
python train.py --load_config configs/backlog/agents20.json --exp_name backlog_agents20 --seed 5

python train.py --load_config configs/backlog/agents30.json --exp_name backlog_agents30 --seed 1
python train.py --load_config configs/backlog/agents30.json --exp_name backlog_agents30 --seed 2
python train.py --load_config configs/backlog/agents30.json --exp_name backlog_agents30 --seed 3
python train.py --load_config configs/backlog/agents30.json --exp_name backlog_agents30 --seed 4
python train.py --load_config configs/backlog/agents30.json --exp_name backlog_agents30 --seed 5

python train.py --load_config configs/backlog/agents50.json --exp_name backlog_agents50 --seed 1
python train.py --load_config configs/backlog/agents50.json --exp_name backlog_agents50 --seed 2
python train.py --load_config configs/backlog/agents50.json --exp_name backlog_agents50 --seed 3
python train.py --load_config configs/backlog/agents50.json --exp_name backlog_agents50 --seed 4
python train.py --load_config configs/backlog/agents50.json --exp_name backlog_agents50 --seed 5

python train.py --load_config configs/backlog/agents6.json --exp_name backlog_agents6 --seed 1
python train.py --load_config configs/backlog/agents6.json --exp_name backlog_agents6 --seed 2
python train.py --load_config configs/backlog/agents6.json --exp_name backlog_agents6 --seed 3
python train.py --load_config configs/backlog/agents6.json --exp_name backlog_agents6 --seed 4
python train.py --load_config configs/backlog/agents6.json --exp_name backlog_agents6 --seed 5

python train.py --load_config configs/backlog/agents9.json --exp_name backlog_agents9 --seed 1
python train.py --load_config configs/backlog/agents9.json --exp_name backlog_agents9 --seed 2
python train.py --load_config configs/backlog/agents9.json --exp_name backlog_agents9 --seed 3
python train.py --load_config configs/backlog/agents9.json --exp_name backlog_agents9 --seed 4
python train.py --load_config configs/backlog/agents9.json --exp_name backlog_agents9 --seed 5

python train.py --load_config configs/backlog/lt3.json --exp_name backlog_lt3 --seed 1
python train.py --load_config configs/backlog/lt3.json --exp_name backlog_lt3 --seed 2
python train.py --load_config configs/backlog/lt3.json --exp_name backlog_lt3 --seed 3
python train.py --load_config configs/backlog/lt3.json --exp_name backlog_lt3 --seed 4
python train.py --load_config configs/backlog/lt3.json --exp_name backlog_lt3 --seed 5

python train.py --load_config configs/backlog/lt4.json --exp_name backlog_lt4 --seed 1
python train.py --load_config configs/backlog/lt4.json --exp_name backlog_lt4 --seed 2
python train.py --load_config configs/backlog/lt4.json --exp_name backlog_lt4 --seed 3
python train.py --load_config configs/backlog/lt4.json --exp_name backlog_lt4 --seed 4
python train.py --load_config configs/backlog/lt4.json --exp_name backlog_lt4 --seed 5

python train.py --load_config configs/backlog/lt5.json --exp_name backlog_lt5 --seed 1
python train.py --load_config configs/backlog/lt5.json --exp_name backlog_lt5 --seed 2
python train.py --load_config configs/backlog/lt5.json --exp_name backlog_lt5 --seed 3
python train.py --load_config configs/backlog/lt5.json --exp_name backlog_lt5 --seed 4
python train.py --load_config configs/backlog/lt5.json --exp_name backlog_lt5 --seed 5

python train.py --load_config configs/backlog/scpu0.json --exp_name backlog_scpu0 --seed 1
python train.py --load_config configs/backlog/scpu0.json --exp_name backlog_scpu0 --seed 2
python train.py --load_config configs/backlog/scpu0.json --exp_name backlog_scpu0 --seed 3
python train.py --load_config configs/backlog/scpu0.json --exp_name backlog_scpu0 --seed 4
python train.py --load_config configs/backlog/scpu0.json --exp_name backlog_scpu0 --seed 5

python train.py --load_config configs/backlog/scpu01.json --exp_name backlog_scpu01 --seed 1
python train.py --load_config configs/backlog/scpu01.json --exp_name backlog_scpu01 --seed 2
python train.py --load_config configs/backlog/scpu01.json --exp_name backlog_scpu01 --seed 3
python train.py --load_config configs/backlog/scpu01.json --exp_name backlog_scpu01 --seed 4
python train.py --load_config configs/backlog/scpu01.json --exp_name backlog_scpu01 --seed 5

python train.py --load_config configs/backlog/scpu03.json --exp_name backlog_scpu03 --seed 1
python train.py --load_config configs/backlog/scpu03.json --exp_name backlog_scpu03 --seed 2
python train.py --load_config configs/backlog/scpu03.json --exp_name backlog_scpu03 --seed 3
python train.py --load_config configs/backlog/scpu03.json --exp_name backlog_scpu03 --seed 4
python train.py --load_config configs/backlog/scpu03.json --exp_name backlog_scpu03 --seed 5

python train.py --load_config configs/backlog/b05.json --exp_name backlog_b05 --seed 1
python train.py --load_config configs/backlog/b05.json --exp_name backlog_b05 --seed 2
python train.py --load_config configs/backlog/b05.json --exp_name backlog_b05 --seed 3
python train.py --load_config configs/backlog/b05.json --exp_name backlog_b05 --seed 4
python train.py --load_config configs/backlog/b05.json --exp_name backlog_b05 --seed 5

python train.py --load_config configs/backlog/b15.json --exp_name backlog_b15 --seed 1
python train.py --load_config configs/backlog/b15.json --exp_name backlog_b15 --seed 2
python train.py --load_config configs/backlog/b15.json --exp_name backlog_b15 --seed 3
python train.py --load_config configs/backlog/b15.json --exp_name backlog_b15 --seed 4
python train.py --load_config configs/backlog/b15.json --exp_name backlog_b15 --seed 5

python train.py --load_config configs/backlog/b20.json --exp_name backlog_b20 --seed 1
python train.py --load_config configs/backlog/b20.json --exp_name backlog_b20 --seed 2
python train.py --load_config configs/backlog/b20.json --exp_name backlog_b20 --seed 3
python train.py --load_config configs/backlog/b20.json --exp_name backlog_b20 --seed 4
python train.py --load_config configs/backlog/b20.json --exp_name backlog_b20 --seed 5

python train.py --load_config configs/backlog/h005.json --exp_name backlog_h005 --seed 1
python train.py --load_config configs/backlog/h005.json --exp_name backlog_h005 --seed 2
python train.py --load_config configs/backlog/h005.json --exp_name backlog_h005 --seed 3
python train.py --load_config configs/backlog/h005.json --exp_name backlog_h005 --seed 4
python train.py --load_config configs/backlog/h005.json --exp_name backlog_h005 --seed 5

python train.py --load_config configs/backlog/h015.json --exp_name backlog_h015 --seed 1
python train.py --load_config configs/backlog/h015.json --exp_name backlog_h015 --seed 2
python train.py --load_config configs/backlog/h015.json --exp_name backlog_h015 --seed 3
python train.py --load_config configs/backlog/h015.json --exp_name backlog_h015 --seed 4
python train.py --load_config configs/backlog/h015.json --exp_name backlog_h015 --seed 5

python train.py --load_config configs/backlog/h020.json --exp_name backlog_h020 --seed 1
python train.py --load_config configs/backlog/h020.json --exp_name backlog_h020 --seed 2
python train.py --load_config configs/backlog/h020.json --exp_name backlog_h020 --seed 3
python train.py --load_config configs/backlog/h020.json --exp_name backlog_h020 --seed 4
python train.py --load_config configs/backlog/h020.json --exp_name backlog_h020 --seed 5

python train.py --load_config configs/backlog/moq0.json --exp_name backlog_moq0 --seed 1
python train.py --load_config configs/backlog/moq0.json --exp_name backlog_moq0 --seed 2
python train.py --load_config configs/backlog/moq0.json --exp_name backlog_moq0 --seed 3
python train.py --load_config configs/backlog/moq0.json --exp_name backlog_moq0 --seed 4
python train.py --load_config configs/backlog/moq0.json --exp_name backlog_moq0 --seed 5

python train.py --load_config configs/backlog/moq40.json --exp_name backlog_moq40 --seed 1
python train.py --load_config configs/backlog/moq40.json --exp_name backlog_moq40 --seed 2
python train.py --load_config configs/backlog/moq40.json --exp_name backlog_moq40 --seed 3
python train.py --load_config configs/backlog/moq40.json --exp_name backlog_moq40 --seed 4
python train.py --load_config configs/backlog/moq40.json --exp_name backlog_moq40 --seed 5

python train.py --load_config configs/backlog/moq60.json --exp_name backlog_moq60 --seed 1
python train.py --load_config configs/backlog/moq60.json --exp_name backlog_moq60 --seed 2
python train.py --load_config configs/backlog/moq60.json --exp_name backlog_moq60 --seed 3
python train.py --load_config configs/backlog/moq60.json --exp_name backlog_moq60 --seed 4
python train.py --load_config configs/backlog/moq60.json --exp_name backlog_moq60 --seed 5
