CUDA_VISIBLE_DEVICES=1 python main.py --data_dir "../data/yelp_restaurant_tag_oov" --data_name "yelp_tag" --vocab_file "sampled_vocab.json" --epoch_num 100 --train --batch_size 400 --learning_rate 0.0001 --print_interval 4000 --weight_decay 0.00000 --attr_emb_size 64 --user_emb_size 64 --item_emb_size 64 --l2_reg 0.0005