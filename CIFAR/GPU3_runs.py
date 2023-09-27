import os

#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 40 > LeNet_SGD_Atan_a0p2b40.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 20 > LeNet_SGD_Atan_a0p2b20.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 10 > LeNet_SGD_Atan_a0p2b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 5 > LeNet_SGD_Atan_a0p2b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 2 > LeNet_SGD_Atan_a0p2b2.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 1 > LeNet_SGD_Atan_a0p2b1.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.5 > LeNet_SGD_Atan_a0p2b0p5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch LeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.25 > LeNet_SGD_Atan_a0p2b0p25.txt 2>&1 "  )

#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch SigmoidLeNet --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 80 > SigmoidLeNet_SGD_Atan_a0p2b80.txt 2>&1 ")

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch SigmoidLeNet --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 40 > SigmoidLeNet_Mom_AtanA0p2b40.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch SigmoidLeNet --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 20 > SigmoidLeNet_Mom_AtanA0p2b20.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 10 > SigmoidLeNet_Mom_AtanA0p2b10.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 5 > SigmoidLeNet_Mom_AtanA0p2b5.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 2 > SigmoidLeNet_Mom_AtanA0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 1 > SigmoidLeNet_Mom_AtanA0p2b1.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.5 > SigmoidLeNet_Mom_AtanA0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.25 > SigmoidLeNet_Mom_AtanA0p2b0p25.txt 2>&1 "  )

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch ResNet50 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 40 > ResNet50_Mom_AtanA0p2b40.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch ResNet50 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 20 > ResNet50_Mom_AtanA0p2b20.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 10 > ResNet50_Mom_AtanA0p2b10.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 5 > ResNet50_Mom_AtanA0p2b5.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 2 > ResNet50_Mom_AtanA0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 1 > ResNet50_Mom_AtanA0p2b1.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.5 > ResNet50_Mom_AtanA0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.25 > ResNet50_Mom_AtanA0p2b0p25.txt 2>&1 "  )

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV1 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 40 > MobileV1_Mom_AtanA0p2b40.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV1 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 20 > MobileV1_Mom_AtanA0p2b20.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 10 > MobileV1_Mom_AtanA0p2b10.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 5 > MobileV1_Mom_AtanA0p2b5.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 2 > MobileV1_Mom_AtanA0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 1 > MobileV1_Mom_AtanA0p2b1.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.5 > MobileV1_Mom_AtanA0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.25 > MobileV1_Mom_AtanA0p2b0p25.txt 2>&1 "  )

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV2 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 40 > MobileV2_Mom_AtanA0p2b40.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV2 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 20 > MobileV2_Mom_AtanA0p2b20.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 10 > MobileV2_Mom_AtanA0p2b10.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 5 > MobileV2_Mom_AtanA0p2b5.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 2 > MobileV2_Mom_AtanA0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 1 > MobileV2_Mom_AtanA0p2b1.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.5 > MobileV2_Mom_AtanA0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.25 > MobileV2_Mom_AtanA0p2b0p25.txt 2>&1 "  )

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch EfficientNetB0 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 40 > EfficientNetB0_Mom_AtanA0p2b40.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch EfficientNetB0 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 20 > EfficientNetB0_Mom_AtanA0p2b20.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 10 > EfficientNetB0_Mom_AtanA0p2b10.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 5 > EfficientNetB0_Mom_AtanA0p2b5.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 2 > EfficientNetB0_Mom_AtanA0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 1 > EfficientNetB0_Mom_AtanA0p2b1.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.5 > EfficientNetB0_Mom_AtanA0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.25 > EfficientNetB0_Mom_AtanA0p2b0p25.txt 2>&1 "  )

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch densenet_cifar --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 40 > densenet_cifar_Mom_AtanA0p2b40.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch densenet_cifar --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 20 > densenet_cifar_Mom_AtanA0p2b20.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 10 > densenet_cifar_Mom_AtanA0p2b10.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 5 > densenet_cifar_Mom_AtanA0p2b5.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 2 > densenet_cifar_Mom_AtanA0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 1 > densenet_cifar_Mom_AtanA0p2b1.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.5 > densenet_cifar_Mom_AtanA0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.25 > densenet_cifar_Mom_AtanA0p2b0p25.txt 2>&1 "  )

os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch RegNetX_200MF --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 40 > RegNetX_200MF_Mom_AtanA0p2b40.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch RegNetX_200MF --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 20 > RegNetX_200MF_Mom_AtanA0p2b20.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 10 > RegNetX_200MF_Mom_AtanA0p2b10.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 5 > RegNetX_200MF_Mom_AtanA0p2b5.txt 2>&1 ") 
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 train.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 2 > RegNetX_200MF_Mom_AtanA0p2b2.txt 2>&1 ") 
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 1 > RegNetX_200MF_Mom_AtanA0p2b1.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.5 > RegNetX_200MF_Mom_AtanA0p2b0p5.txt 2>&1 " )
os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.2  --beta 0.25 > RegNetX_200MF_Mom_AtanA0p2b0p25.txt 2>&1 "  )
 
#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 40 > VGG16_SGD_atanMomAtan_a0p2b40.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 20 > VGG16_SGD_atanMomAtan_a0p2b20.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 10 > VGG16_SGD_atanMomAtan_a0p2b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 5 > VGG16_SGD_atanMomAtan_a0p2b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 2 > VGG16_SGD_atanMomAtan_a0p2b2.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 1 > VGG16_SGD_atanMomAtan_a0p2b1.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.5 > VGG16_SGD_atanMomAtan_a0p2b0p5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch VGG16 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.25 > VGG16_SGD_atanMomAtan_a0p2b0p25.txt 2>&1 "  )


#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 40 > ResNet152_SGD_Atan_a0p2b40.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 20 > ResNet152_SGD_Atan_a0p2b20.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 10 > ResNet152_SGD_Atan_a0p2b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 5 > ResNet152_SGD_Atan_a0p2b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 2 > ResNet152_SGD_Atan_a0p2b2.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 1 > ResNet152_SGD_Atan_a0p2b1.txt 2>&1  " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.5 > ResNet152_SGD_Atan_a0p2b0p5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=3 nohup python3  train.py --arch ResNet152 --opt SGD_atan -bs 256 --lr 0.1 --workers 4 --alpha 0.2 --beta 0.25 > ResNet152_SGD_Atan_a0p2b0p25.txt 2>&1 "  )