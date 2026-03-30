#!/bin/bash

# enable GPU rendering
export GPU_RENDER=1

# start render
python render_toolkit.py \
    --test_list 790 \
    --mesh_dir "path/to/mesh_dir" \
    --data_dir "path/to/data_dir" \
    --cate "category" \
    --test_path "path/to/test_path"

