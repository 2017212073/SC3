#!/bin/bash

# 激活 Torch 环境
source activate torch_xzt

yaml_file="config/config.yaml"
old_file="save/XSRP_task10_HCNN/result_record.txt"
for seed in $(seq 40 49); do
    # 使用 yq 更新 seed 值
    echo $seed
    sed -i "s/seed: [0-9]*/seed: $seed/" "$yaml_file"
    python train.py
    python cluster.py
    new_file="${old_file%.txt}_$seed.txt"
    mv "$old_file" "$new_file"
done
# 依次运行 Python 脚本


echo "所有脚本顺利运行完成！"
