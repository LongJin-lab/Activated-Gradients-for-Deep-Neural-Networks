import os


os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet152 --opt SGD_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > LeNet_SGD_ori.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet152 --opt SGD_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > ResNet152_SGD_ori.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > ResNet152_AtanA0p02b400.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > ResNet152_AtanA0p02b200.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > ResNet152_AtanA0p02b100.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > ResNet152_AtanA0p02b50.txt 2>&1 ") 

os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > ResNet152_AtanA0p02b25.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=0 nohup python3  train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > ResNet152_AtanA0p02b10.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=0 nohup python3  train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > ResNet152_AtanA0p02b5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=0 nohup python3  train.py --arch ResNet152 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > ResNet152_AtanA0p02b5.txt 2>&1 "  )




#os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch ResNet18 --opt Adam_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > ResNet18_Adam_ori.txt 2>&1 ") 
#
#
#os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch VGG16 --opt Adam -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > VGG16_Adam_ori.txt 2>&1 ") 
#
#os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > VGG16_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > VGG16_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > VGG16_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > VGG16_AtanA0p02b50.txt 2>&1 ") 
#
#os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > VGG16_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=0 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > VGG16_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=0 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > VGG16_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=0 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > VGG16_AtanA0p02b5.txt 2>&1 "  )

