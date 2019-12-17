nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_base.yaml --run-type train \
--gpu-id 0 --experiment base5_base > results/12_dec_base_5step/base5_base.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_depth.yaml --run-type train \
--gpu-id 1 --experiment base5_depth > results/12_dec_base_5step/base5_depth.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_drop.yaml --run-type train \
--gpu-id 1 --experiment base5_drop > results/12_dec_base_5step/base5_drop.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_batch3.yaml --run-type train \
--gpu-id 2 --experiment base5_batch3 > results/12_dec_base_5step/base5_batch3.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_noisy02.yaml --run-type train \
--gpu-id 2 --experiment base5_noisy02 > results/12_dec_base_5step/base5_noisy02.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_noisy03.yaml --run-type train \
--gpu-id 3 --experiment base5_noisy03 > results/12_dec_base_5step/base5_noisy03.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_noop.yaml --run-type train \
--gpu-id 3 --experiment base5_noop > results/12_dec_base_5step/base5_noop.out 2>&1 &

# Not done

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_noisy01.yaml --run-type train \
--gpu-id 1 --experiment base5_noisy01 > results/12_dec_base_5step/base5_noisy01.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_rgb.yaml --run-type train \
--gpu-id 2 --experiment base5_rgb > results/12_dec_base_5step/base5_rgb.out 2>&1 &

nohup python -u habitat_baselines/run.py --exp-config \
habitat_baselines/config/visual_aug_multistep/base_tilt.yaml --run-type train \
--gpu-id 3 --experiment base5_tilt > results/12_dec_base_5step/base5_tilt.out 2>&1 &


