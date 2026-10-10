#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python train.py --load_config configs/ririshun/1936c558.json --exp_name ririshun_1936c558 --seed 1
python train.py --load_config configs/ririshun/1936c558.json --exp_name ririshun_1936c558 --seed 2
python train.py --load_config configs/ririshun/1936c558.json --exp_name ririshun_1936c558 --seed 3
python train.py --load_config configs/ririshun/1936c558.json --exp_name ririshun_1936c558 --seed 4
python train.py --load_config configs/ririshun/1936c558.json --exp_name ririshun_1936c558 --seed 5

python train.py --load_config configs/ririshun/26f47370.json --exp_name ririshun_26f47370 --seed 1
python train.py --load_config configs/ririshun/26f47370.json --exp_name ririshun_26f47370 --seed 2
python train.py --load_config configs/ririshun/26f47370.json --exp_name ririshun_26f47370 --seed 3
python train.py --load_config configs/ririshun/26f47370.json --exp_name ririshun_26f47370 --seed 4
python train.py --load_config configs/ririshun/26f47370.json --exp_name ririshun_26f47370 --seed 5

python train.py --load_config configs/ririshun/3b6fda08.json --exp_name ririshun_3b6fda08 --seed 1
python train.py --load_config configs/ririshun/3b6fda08.json --exp_name ririshun_3b6fda08 --seed 2
python train.py --load_config configs/ririshun/3b6fda08.json --exp_name ririshun_3b6fda08 --seed 3
python train.py --load_config configs/ririshun/3b6fda08.json --exp_name ririshun_3b6fda08 --seed 4
python train.py --load_config configs/ririshun/3b6fda08.json --exp_name ririshun_3b6fda08 --seed 5

python train.py --load_config configs/ririshun/3e0092db.json --exp_name ririshun_3e0092db --seed 1
python train.py --load_config configs/ririshun/3e0092db.json --exp_name ririshun_3e0092db --seed 2
python train.py --load_config configs/ririshun/3e0092db.json --exp_name ririshun_3e0092db --seed 3
python train.py --load_config configs/ririshun/3e0092db.json --exp_name ririshun_3e0092db --seed 4
python train.py --load_config configs/ririshun/3e0092db.json --exp_name ririshun_3e0092db --seed 5

python train.py --load_config configs/ririshun/a4830d20.json --exp_name ririshun_a4830d20 --seed 1
python train.py --load_config configs/ririshun/a4830d20.json --exp_name ririshun_a4830d20 --seed 2
python train.py --load_config configs/ririshun/a4830d20.json --exp_name ririshun_a4830d20 --seed 3
python train.py --load_config configs/ririshun/a4830d20.json --exp_name ririshun_a4830d20 --seed 4
python train.py --load_config configs/ririshun/a4830d20.json --exp_name ririshun_a4830d20 --seed 5

python train.py --load_config configs/ririshun/b01ff11b.json --exp_name ririshun_b01ff11b --seed 1
python train.py --load_config configs/ririshun/b01ff11b.json --exp_name ririshun_b01ff11b --seed 2
python train.py --load_config configs/ririshun/b01ff11b.json --exp_name ririshun_b01ff11b --seed 3
python train.py --load_config configs/ririshun/b01ff11b.json --exp_name ririshun_b01ff11b --seed 4
python train.py --load_config configs/ririshun/b01ff11b.json --exp_name ririshun_b01ff11b --seed 5

python train.py --load_config configs/ririshun/b0bdce95.json --exp_name ririshun_b0bdce95 --seed 1
python train.py --load_config configs/ririshun/b0bdce95.json --exp_name ririshun_b0bdce95 --seed 2
python train.py --load_config configs/ririshun/b0bdce95.json --exp_name ririshun_b0bdce95 --seed 3
python train.py --load_config configs/ririshun/b0bdce95.json --exp_name ririshun_b0bdce95 --seed 4
python train.py --load_config configs/ririshun/b0bdce95.json --exp_name ririshun_b0bdce95 --seed 5

python train.py --load_config configs/ririshun/bf67f87f.json --exp_name ririshun_bf67f87f --seed 1
python train.py --load_config configs/ririshun/bf67f87f.json --exp_name ririshun_bf67f87f --seed 2
python train.py --load_config configs/ririshun/bf67f87f.json --exp_name ririshun_bf67f87f --seed 3
python train.py --load_config configs/ririshun/bf67f87f.json --exp_name ririshun_bf67f87f --seed 4
python train.py --load_config configs/ririshun/bf67f87f.json --exp_name ririshun_bf67f87f --seed 5
