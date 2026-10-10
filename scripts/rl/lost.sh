#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python train.py --load_config configs/lost/default.json --exp_name lost_default --seed 1
python train.py --load_config configs/lost/default.json --exp_name lost_default --seed 2
python train.py --load_config configs/lost/default.json --exp_name lost_default --seed 3
python train.py --load_config configs/lost/default.json --exp_name lost_default --seed 4
python train.py --load_config configs/lost/default.json --exp_name lost_default --seed 5

python train.py --load_config configs/lost/agents12.json --exp_name lost_agents12 --seed 1
python train.py --load_config configs/lost/agents12.json --exp_name lost_agents12 --seed 2
python train.py --load_config configs/lost/agents12.json --exp_name lost_agents12 --seed 3
python train.py --load_config configs/lost/agents12.json --exp_name lost_agents12 --seed 4
python train.py --load_config configs/lost/agents12.json --exp_name lost_agents12 --seed 5

python train.py --load_config configs/lost/agents15.json --exp_name lost_agents15 --seed 1
python train.py --load_config configs/lost/agents15.json --exp_name lost_agents15 --seed 2
python train.py --load_config configs/lost/agents15.json --exp_name lost_agents15 --seed 3
python train.py --load_config configs/lost/agents15.json --exp_name lost_agents15 --seed 4
python train.py --load_config configs/lost/agents15.json --exp_name lost_agents15 --seed 5

python train.py --load_config configs/lost/agents2.json --exp_name lost_agents2 --seed 1
python train.py --load_config configs/lost/agents2.json --exp_name lost_agents2 --seed 2
python train.py --load_config configs/lost/agents2.json --exp_name lost_agents2 --seed 3
python train.py --load_config configs/lost/agents2.json --exp_name lost_agents2 --seed 4
python train.py --load_config configs/lost/agents2.json --exp_name lost_agents2 --seed 5

python train.py --load_config configs/lost/agents20.json --exp_name lost_agents20 --seed 1
python train.py --load_config configs/lost/agents20.json --exp_name lost_agents20 --seed 2
python train.py --load_config configs/lost/agents20.json --exp_name lost_agents20 --seed 3
python train.py --load_config configs/lost/agents20.json --exp_name lost_agents20 --seed 4
python train.py --load_config configs/lost/agents20.json --exp_name lost_agents20 --seed 5

python train.py --load_config configs/lost/agents30.json --exp_name lost_agents30 --seed 1
python train.py --load_config configs/lost/agents30.json --exp_name lost_agents30 --seed 2
python train.py --load_config configs/lost/agents30.json --exp_name lost_agents30 --seed 3
python train.py --load_config configs/lost/agents30.json --exp_name lost_agents30 --seed 4
python train.py --load_config configs/lost/agents30.json --exp_name lost_agents30 --seed 5

python train.py --load_config configs/lost/agents50.json --exp_name lost_agents50 --seed 1
python train.py --load_config configs/lost/agents50.json --exp_name lost_agents50 --seed 2
python train.py --load_config configs/lost/agents50.json --exp_name lost_agents50 --seed 3
python train.py --load_config configs/lost/agents50.json --exp_name lost_agents50 --seed 4
python train.py --load_config configs/lost/agents50.json --exp_name lost_agents50 --seed 5

python train.py --load_config configs/lost/agents6.json --exp_name lost_agents6 --seed 1
python train.py --load_config configs/lost/agents6.json --exp_name lost_agents6 --seed 2
python train.py --load_config configs/lost/agents6.json --exp_name lost_agents6 --seed 3
python train.py --load_config configs/lost/agents6.json --exp_name lost_agents6 --seed 4
python train.py --load_config configs/lost/agents6.json --exp_name lost_agents6 --seed 5

python train.py --load_config configs/lost/agents9.json --exp_name lost_agents9 --seed 1
python train.py --load_config configs/lost/agents9.json --exp_name lost_agents9 --seed 2
python train.py --load_config configs/lost/agents9.json --exp_name lost_agents9 --seed 3
python train.py --load_config configs/lost/agents9.json --exp_name lost_agents9 --seed 4
python train.py --load_config configs/lost/agents9.json --exp_name lost_agents9 --seed 5

python train.py --load_config configs/lost/lt3.json --exp_name lost_lt3 --seed 1
python train.py --load_config configs/lost/lt3.json --exp_name lost_lt3 --seed 2
python train.py --load_config configs/lost/lt3.json --exp_name lost_lt3 --seed 3
python train.py --load_config configs/lost/lt3.json --exp_name lost_lt3 --seed 4
python train.py --load_config configs/lost/lt3.json --exp_name lost_lt3 --seed 5

python train.py --load_config configs/lost/lt4.json --exp_name lost_lt4 --seed 1
python train.py --load_config configs/lost/lt4.json --exp_name lost_lt4 --seed 2
python train.py --load_config configs/lost/lt4.json --exp_name lost_lt4 --seed 3
python train.py --load_config configs/lost/lt4.json --exp_name lost_lt4 --seed 4
python train.py --load_config configs/lost/lt4.json --exp_name lost_lt4 --seed 5

python train.py --load_config configs/lost/lt5.json --exp_name lost_lt5 --seed 1
python train.py --load_config configs/lost/lt5.json --exp_name lost_lt5 --seed 2
python train.py --load_config configs/lost/lt5.json --exp_name lost_lt5 --seed 3
python train.py --load_config configs/lost/lt5.json --exp_name lost_lt5 --seed 4
python train.py --load_config configs/lost/lt5.json --exp_name lost_lt5 --seed 5

python train.py --load_config configs/lost/scpu0.json --exp_name lost_scpu0 --seed 1
python train.py --load_config configs/lost/scpu0.json --exp_name lost_scpu0 --seed 2
python train.py --load_config configs/lost/scpu0.json --exp_name lost_scpu0 --seed 3
python train.py --load_config configs/lost/scpu0.json --exp_name lost_scpu0 --seed 4
python train.py --load_config configs/lost/scpu0.json --exp_name lost_scpu0 --seed 5

python train.py --load_config configs/lost/scpu01.json --exp_name lost_scpu01 --seed 1
python train.py --load_config configs/lost/scpu01.json --exp_name lost_scpu01 --seed 2
python train.py --load_config configs/lost/scpu01.json --exp_name lost_scpu01 --seed 3
python train.py --load_config configs/lost/scpu01.json --exp_name lost_scpu01 --seed 4
python train.py --load_config configs/lost/scpu01.json --exp_name lost_scpu01 --seed 5

python train.py --load_config configs/lost/scpu03.json --exp_name lost_scpu03 --seed 1
python train.py --load_config configs/lost/scpu03.json --exp_name lost_scpu03 --seed 2
python train.py --load_config configs/lost/scpu03.json --exp_name lost_scpu03 --seed 3
python train.py --load_config configs/lost/scpu03.json --exp_name lost_scpu03 --seed 4
python train.py --load_config configs/lost/scpu03.json --exp_name lost_scpu03 --seed 5

python train.py --load_config configs/lost/b05.json --exp_name lost_b05 --seed 1
python train.py --load_config configs/lost/b05.json --exp_name lost_b05 --seed 2
python train.py --load_config configs/lost/b05.json --exp_name lost_b05 --seed 3
python train.py --load_config configs/lost/b05.json --exp_name lost_b05 --seed 4
python train.py --load_config configs/lost/b05.json --exp_name lost_b05 --seed 5

python train.py --load_config configs/lost/b15.json --exp_name lost_b15 --seed 1
python train.py --load_config configs/lost/b15.json --exp_name lost_b15 --seed 2
python train.py --load_config configs/lost/b15.json --exp_name lost_b15 --seed 3
python train.py --load_config configs/lost/b15.json --exp_name lost_b15 --seed 4
python train.py --load_config configs/lost/b15.json --exp_name lost_b15 --seed 5

python train.py --load_config configs/lost/b20.json --exp_name lost_b20 --seed 1
python train.py --load_config configs/lost/b20.json --exp_name lost_b20 --seed 2
python train.py --load_config configs/lost/b20.json --exp_name lost_b20 --seed 3
python train.py --load_config configs/lost/b20.json --exp_name lost_b20 --seed 4
python train.py --load_config configs/lost/b20.json --exp_name lost_b20 --seed 5

python train.py --load_config configs/lost/h005.json --exp_name lost_h005 --seed 1
python train.py --load_config configs/lost/h005.json --exp_name lost_h005 --seed 2
python train.py --load_config configs/lost/h005.json --exp_name lost_h005 --seed 3
python train.py --load_config configs/lost/h005.json --exp_name lost_h005 --seed 4
python train.py --load_config configs/lost/h005.json --exp_name lost_h005 --seed 5

python train.py --load_config configs/lost/h015.json --exp_name lost_h015 --seed 1
python train.py --load_config configs/lost/h015.json --exp_name lost_h015 --seed 2
python train.py --load_config configs/lost/h015.json --exp_name lost_h015 --seed 3
python train.py --load_config configs/lost/h015.json --exp_name lost_h015 --seed 4
python train.py --load_config configs/lost/h015.json --exp_name lost_h015 --seed 5

python train.py --load_config configs/lost/h020.json --exp_name lost_h020 --seed 1
python train.py --load_config configs/lost/h020.json --exp_name lost_h020 --seed 2
python train.py --load_config configs/lost/h020.json --exp_name lost_h020 --seed 3
python train.py --load_config configs/lost/h020.json --exp_name lost_h020 --seed 4
python train.py --load_config configs/lost/h020.json --exp_name lost_h020 --seed 5

python train.py --load_config configs/lost/moq0.json --exp_name lost_moq0 --seed 1
python train.py --load_config configs/lost/moq0.json --exp_name lost_moq0 --seed 2
python train.py --load_config configs/lost/moq0.json --exp_name lost_moq0 --seed 3
python train.py --load_config configs/lost/moq0.json --exp_name lost_moq0 --seed 4
python train.py --load_config configs/lost/moq0.json --exp_name lost_moq0 --seed 5

python train.py --load_config configs/lost/moq40.json --exp_name lost_moq40 --seed 1
python train.py --load_config configs/lost/moq40.json --exp_name lost_moq40 --seed 2
python train.py --load_config configs/lost/moq40.json --exp_name lost_moq40 --seed 3
python train.py --load_config configs/lost/moq40.json --exp_name lost_moq40 --seed 4
python train.py --load_config configs/lost/moq40.json --exp_name lost_moq40 --seed 5

python train.py --load_config configs/lost/moq60.json --exp_name lost_moq60 --seed 1
python train.py --load_config configs/lost/moq60.json --exp_name lost_moq60 --seed 2
python train.py --load_config configs/lost/moq60.json --exp_name lost_moq60 --seed 3
python train.py --load_config configs/lost/moq60.json --exp_name lost_moq60 --seed 4
python train.py --load_config configs/lost/moq60.json --exp_name lost_moq60 --seed 5

python train.py --load_config configs/lost/p4.json --exp_name lost_p4 --seed 1
python train.py --load_config configs/lost/p4.json --exp_name lost_p4 --seed 2
python train.py --load_config configs/lost/p4.json --exp_name lost_p4 --seed 3
python train.py --load_config configs/lost/p4.json --exp_name lost_p4 --seed 4
python train.py --load_config configs/lost/p4.json --exp_name lost_p4 --seed 5

python train.py --load_config configs/lost/p6.json --exp_name lost_p6 --seed 1
python train.py --load_config configs/lost/p6.json --exp_name lost_p6 --seed 2
python train.py --load_config configs/lost/p6.json --exp_name lost_p6 --seed 3
python train.py --load_config configs/lost/p6.json --exp_name lost_p6 --seed 4
python train.py --load_config configs/lost/p6.json --exp_name lost_p6 --seed 5

python train.py --load_config configs/lost/p7.json --exp_name lost_p7 --seed 1
python train.py --load_config configs/lost/p7.json --exp_name lost_p7 --seed 2
python train.py --load_config configs/lost/p7.json --exp_name lost_p7 --seed 3
python train.py --load_config configs/lost/p7.json --exp_name lost_p7 --seed 4
python train.py --load_config configs/lost/p7.json --exp_name lost_p7 --seed 5
