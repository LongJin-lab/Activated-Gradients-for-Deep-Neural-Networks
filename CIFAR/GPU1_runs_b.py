import os

os.system("CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 80 > ResNet152_SGD_Atan_a0p05b80.txt 2>&1  ") 
os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 60 > ResNet152_SGD_Atan_a0p05b60.txt 2>&1  " )
os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 40 > ResNet152_SGD_Atan_a0p05b40.txt 2>&1  " )
os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 20 > ResNet152_SGD_Atan_a0p05b20.txt 2>&1  "  )

os.system("CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 10 > ResNet152_SGD_Atan_a0p05b10.txt 2>&1  ") 
os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 5 > ResNet152_SGD_Atan_a0p05b5.txt 2>&1  " )
os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 2 > ResNet152_SGD_Atan_a0p05b2.txt 2>&1  " )
os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 1 > ResNet152_SGD_Atan_a0p05b1.txt 2>&1  "  )

#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 80 > VGG16_Atan_a0p05b80.txt 2>&1  ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 60 > VGG16_Atan_a0p05b60.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 40 > VGG16_Atan_a0p05b40.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 20 > VGG16_Atan_a0p05b20.txt 2>&1  "  )
#
#
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 10 > VGG16_Atan_a0p05b10.txt 2>&1  ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 5 > VGG16_Atan_a0p05b5.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 2 > VGG16_Atan_a0p05b2.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.05 --beta 1 > VGG16_Atan_a0p05b1.txt 2>&1  "  )

