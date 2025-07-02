dataset=Yelp

python ./RQ-VAE/generate_indices.py\
    --dataset ${dataset} \
    --root_path ./checkpoint/${dataset}/ \
    --alpha 0.01 \
    --beta 0.0001 \
    --epoch 20000 \
    --checkpoint best_collision_model.pth 