#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python train.py --load_config configs/beverage/SKU006.json --exp_name beverage_SKU006 --seed 1
python train.py --load_config configs/beverage/SKU006.json --exp_name beverage_SKU006 --seed 2
python train.py --load_config configs/beverage/SKU006.json --exp_name beverage_SKU006 --seed 3
python train.py --load_config configs/beverage/SKU006.json --exp_name beverage_SKU006 --seed 4
python train.py --load_config configs/beverage/SKU006.json --exp_name beverage_SKU006 --seed 5

python train.py --load_config configs/beverage/SKU019.json --exp_name beverage_SKU019 --seed 1
python train.py --load_config configs/beverage/SKU019.json --exp_name beverage_SKU019 --seed 2
python train.py --load_config configs/beverage/SKU019.json --exp_name beverage_SKU019 --seed 3
python train.py --load_config configs/beverage/SKU019.json --exp_name beverage_SKU019 --seed 4
python train.py --load_config configs/beverage/SKU019.json --exp_name beverage_SKU019 --seed 5

python train.py --load_config configs/beverage/SKU022.json --exp_name beverage_SKU022 --seed 1
python train.py --load_config configs/beverage/SKU022.json --exp_name beverage_SKU022 --seed 2
python train.py --load_config configs/beverage/SKU022.json --exp_name beverage_SKU022 --seed 3
python train.py --load_config configs/beverage/SKU022.json --exp_name beverage_SKU022 --seed 4
python train.py --load_config configs/beverage/SKU022.json --exp_name beverage_SKU022 --seed 5

python train.py --load_config configs/beverage/SKU023.json --exp_name beverage_SKU023 --seed 1
python train.py --load_config configs/beverage/SKU023.json --exp_name beverage_SKU023 --seed 2
python train.py --load_config configs/beverage/SKU023.json --exp_name beverage_SKU023 --seed 3
python train.py --load_config configs/beverage/SKU023.json --exp_name beverage_SKU023 --seed 4
python train.py --load_config configs/beverage/SKU023.json --exp_name beverage_SKU023 --seed 5

python train.py --load_config configs/beverage/SKU029.json --exp_name beverage_SKU029 --seed 1
python train.py --load_config configs/beverage/SKU029.json --exp_name beverage_SKU029 --seed 2
python train.py --load_config configs/beverage/SKU029.json --exp_name beverage_SKU029 --seed 3
python train.py --load_config configs/beverage/SKU029.json --exp_name beverage_SKU029 --seed 4
python train.py --load_config configs/beverage/SKU029.json --exp_name beverage_SKU029 --seed 5

python train.py --load_config configs/beverage/SKU045.json --exp_name beverage_SKU045 --seed 1
python train.py --load_config configs/beverage/SKU045.json --exp_name beverage_SKU045 --seed 2
python train.py --load_config configs/beverage/SKU045.json --exp_name beverage_SKU045 --seed 3
python train.py --load_config configs/beverage/SKU045.json --exp_name beverage_SKU045 --seed 4
python train.py --load_config configs/beverage/SKU045.json --exp_name beverage_SKU045 --seed 5

python train.py --load_config configs/beverage/SKU046.json --exp_name beverage_SKU046 --seed 1
python train.py --load_config configs/beverage/SKU046.json --exp_name beverage_SKU046 --seed 2
python train.py --load_config configs/beverage/SKU046.json --exp_name beverage_SKU046 --seed 3
python train.py --load_config configs/beverage/SKU046.json --exp_name beverage_SKU046 --seed 4
python train.py --load_config configs/beverage/SKU046.json --exp_name beverage_SKU046 --seed 5

python train.py --load_config configs/beverage/SKU062.json --exp_name beverage_SKU062 --seed 1
python train.py --load_config configs/beverage/SKU062.json --exp_name beverage_SKU062 --seed 2
python train.py --load_config configs/beverage/SKU062.json --exp_name beverage_SKU062 --seed 3
python train.py --load_config configs/beverage/SKU062.json --exp_name beverage_SKU062 --seed 4
python train.py --load_config configs/beverage/SKU062.json --exp_name beverage_SKU062 --seed 5
