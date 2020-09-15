python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr="128.143.67.123" --master_port=1234 main.py --data_dir "../data/yelp_restaurant_tag_oov" --train --vocab_file "vocab.json" --data_name "yelp_tag" --parallel --epoch_num 100 --batch_size 400 --learning_rate 0.000001 --print_interval 10000 --weight_decay 0.00000 --attr_emb_size 128 --user_emb_size 128 --item_emb_size 128