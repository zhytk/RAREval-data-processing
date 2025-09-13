#!/bin/bash
python train_valid_test_split.py --json_input_file Musical_Instruments_5.json --save_directory reviews_Musical_Instruments
python train_valid_test_split.py --json_input_file Amazon_Instant_Video_5.json --save_directory reviews_Amazon_Instant_Video
python train_valid_test_split.py --json_input_file Digital_Music_5.json --save_directory reviews_Digital_Music
python train_valid_test_split.py --json_input_file Video_Games_5.json --save_directory reviews_Video_Games
python train_valid_test_split.py --json_input_file Office_Products_5.json --save_directory reviews_Office_Products
python train_valid_test_split.py --json_input_file Health_and_Personal_Care_5.json --save_directory reviews_Health_and_Personal_Care
python train_valid_test_split.py --json_input_file CDs_and_Vinyl_5.json --save_directory reviews_CDs_and_Vinyl
python train_valid_test_split.py --json_input_file Movies_and_TV_5.json --save_directory reviews_Movies_and_TV
