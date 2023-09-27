import os

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 40 > ResNet152_SGD_Atan_a0p2b40.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 20 > ResNet152_SGD_Atan_a0p2b20.txt 2>&1  " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 10 > ResNet152_SGD_Atan_a0p2b10.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 5 > ResNet152_SGD_Atan_a0p2b5.txt 2>&1 "  )

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 2 > ResNet152_SGD_Atan_a0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 1 > ResNet152_SGD_Atan_a0p2b1.txt 2>&1  " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.5 > ResNet152_SGD_Atan_a0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.25 > ResNet152_SGD_Atan_a0p2b0p25.txt 2>&1 "  )


#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 40 > VGG16_Atan_a0p2b40.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 20 > VGG16_Atan_a0p2b20.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 10 > VGG16_Atan_a0p2b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 5 > VGG16_Atan_a0p2b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 2 > VGG16_Atan_a0p2b2.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 1 > VGG16_Atan_a0p2b1.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.5 > VGG16_Atan_a0p2b0p5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.25 > VGG16_Atan_a0p2b0p25.txt 2>&1 "  )

