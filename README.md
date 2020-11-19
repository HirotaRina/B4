#学習環境を整える
$ python create_input_files.py
#学習
$ python cubTrain.py
$ pip uninstall scipy
$ pip install scipy==1.2.0
$ pip uninstall pillow
$ pip install pillow==6.2.1
＃識別
$ python caption.py  --img='/mnt/exthd1/cub/CUB_200_2011/images/065.Slaty_backed_Gull/Slaty_Backed_Gull_0081_53264.jpg' --model='./BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --beam_size=5
# B4
