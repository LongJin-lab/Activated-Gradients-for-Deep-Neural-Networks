import os
import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'
#model_names_cifar100 = ['MobileV1', 'MobileV2', 'ResNet152', 'ResNet18', 'ResNet34', 'ResNet101', 'EfficientNetB0', 'densenet_cifar', 'VGG16', 'ResNet50', 'RegNetX_200MF', 'SigmoidLeNet', 'ShuffleNetV2', 'GoogLeNet', 'ResNeXt29_2x64d', 'SENet18', 'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'DPN26', 'ShuffleNetG2','DenseNet121', 'DenseNet169']
model_names_cifar100 = ['ResNet18', 'ResNet34', 'ResNet101', 'EfficientNetB0', 'densenet_cifar', 'VGG16', 'ResNet50', 'SigmoidLeNet', 'ShuffleNetV2', 'GoogLeNet', 'SENet18', 'PreActResNet18', 'PreActResNet34', 'PreActResNet50','DenseNet121']

#model_names_cifar10 = [ 'ResNet34', 'ResNet101', 'ShuffleNetV2', 'GoogLeNet', 'ResNeXt29_2x64d', 'SENet18', 'PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'DPN26', 'ShuffleNetG2','DenseNet121', 'DenseNet169']

model_names_cifar10 = model_names_cifar100
alpha2 = -1
beta2 = -1
i = 0
optimizer  = 'Adam_ori'       
for model_name_cifar100 in model_names_cifar100:
    
    if i % 8 == 1:
        print('model_name_cifar100_G0', model_name_cifar100)   

        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar100.py --arch "+ model_name_cifar100+" --dataset cifar100 --opt " + optimizer +" -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar100+"_c100_Adam_ori_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "
        os.system(command)
    i = i+1 
i = 0    
for model_name_cifar10 in model_names_cifar10:
    
    if i % 8 == 1:
        print('model_name_cifar10_G0', model_name_cifar10)   

        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name_cifar10+" --dataset cifar10 --opt " + optimizer +" -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar10+"_c10_Adam_ori_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "    
        os.system(command)
    i = i+1 
    
alpha2 = 0.1
beta2 = 20
i = 0
optimizer  = 'Adam_atan'       
for model_name_cifar100 in model_names_cifar100:
    
    if i % 8 == 1:
        print('model_name_cifar100_G0', model_name_cifar100)   

        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar100.py --arch "+ model_name_cifar100+" --dataset cifar100 --opt " + optimizer +" -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar100+"_c100_Adam_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "
        os.system(command)
    i = i+1 
i = 0    
for model_name_cifar10 in model_names_cifar10:
    
    if i % 8 == 1:
        print('model_name_cifar10_G0', model_name_cifar10)   

        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name_cifar10+" --dataset cifar10 --opt " + optimizer +" -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar10+"_c10_Adam_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "    
        os.system(command)
    i = i+1  

alpha2 = 0.2
beta2 = 10
i = 0       
for model_name_cifar100 in model_names_cifar100:
    
    if i % 8 == 1:
        print('model_name_cifar100_G0', model_name_cifar100)   

        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar100.py --arch "+ model_name_cifar100+" --dataset cifar100 --opt " + optimizer +" -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar100+"_c100_Adam_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "
        os.system(command)
    i = i+1 
i = 0    
for model_name_cifar10 in model_names_cifar10:
    
    if i % 8 == 1:
        print('model_name_cifar10_G0', model_name_cifar10)   

        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name_cifar10+" --dataset cifar10 --opt " + optimizer +" -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar10+"_c10_Adam_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "    
        os.system(command)
    i = i+1 

#alpha2 = 0.01
#beta2 = 150
#i = 0       
#for model_name_cifar100 in model_names_cifar100:
#    
#    if i % 4 == 1:
#        print('model_name_cifar100_G0', model_name_cifar100)   
#
#        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar100.py --arch "+ model_name_cifar100+" --dataset cifar100 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar100+"_c100_Adam_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "
#        os.system(command)
#    i = i+1 
#i = 0    
#for model_name_cifar10 in model_names_cifar10:
#    
#    if i % 4 == 1:
#        print('model_name_cifar10_G0', model_name_cifar10)   
#
#        command = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name_cifar10+" --dataset cifar10 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name_cifar10+"_c10_Adam_A"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "    
#        os.system(command)
#    i = i+1     
           
#i = 0
#for model_name_cifar100 in model_names_cifar100:
#    
#    if i % 4 == 2:
#        print('model_name_cifar100_G1', model_name_cifar100)
#    i = i+1
#    
#i = 0
#for model_name_cifar100 in model_names_cifar100:
#    
#    if i % 4 == 0:
#        print('model_name_cifar100_G2', model_name_cifar100)    
#    i = i+1
#    
#i = 0
#for model_name_cifar100 in model_names_cifar100:
#    
#    if i % 4 == 3:
#        print('model_name_cifar100_G3', model_name_cifar100)    
#    i = i+1

    
        
        
#i=0    
#for model_name_cifar10 in model_names_cifar10:
#    i = 1+1
#    if i % 4 == 1:        
#        print('model_name_cifar10', model_name_cifar10)
    
    

##for i in range(0, 7, 2):
##    alpha = 0.1*1.2**(i-3)
#for i in range(0, 11, 2):
#    alpha = 0.1*1.2**(i-5)    
#    beta = 1/alpha
#    if i == 0:
#        alphas = alpha
#        betas = beta
#    else:
#        alphas = np.row_stack((alphas, alpha))
#        betas = np.row_stack((betas, beta))
#    alpha2 = alpha*1.2
#    beta2 = 1/alpha2
#    alphas = np.row_stack((alphas, alpha2))
#    betas = np.row_stack((betas, beta2))
#
#
#        
##    print('alpha, beta', alpha, beta )
##    print('alpha2, beta2', alpha2, beta2 )
##    command1 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha)+" --beta "+ str(beta)+" > "+model_name+"_SGD_atanMomAtanA"+str(alpha)+"b"+str(beta)+".txt 2>&1 &"
##    command2 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name+"_SGD_atanMomAtanA"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "
##    print('command1', command1)
##    print('command2', command2)
##    os.system(command1)
##    os.system(command2)    
#print('alphas, betas', alphas, betas )
#cnt = 0
#for i in alphas:
#    for j in betas:
##        if i[0]*j[0]>=1:
#        if cnt == 0:
#            ABs = [i[0],j[0]]
#        else:
#            ABs = np.row_stack((ABs, [ i[0],j[0] ]))
#        cnt = cnt + 1 
#
#for i in range(0, ABs.shape[0], 4 ):
#    print(i, ABs[i][0], ABs[i][1])
#    print(i+1, ABs[i+1][0], ABs[i+1][1])
#    print(i+2, ABs[i+2][0], ABs[i+2][1])
#    print(i+3, ABs[i+3][0], ABs[i+3][1])      
      
      
#    command1 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i][0])+" --beta "+ str(ABs[i][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i][0])+"b"+str(ABs[i][1])+".txt 2>&1 &"
#    command2 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i+1][0])+" --beta "+ str(ABs[i+1][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i+1][0])+"b"+str(ABs[i+1][1])+".txt 2>&1 &"
#    command3 = "CUDA_VISIBLE_DEVICES=2 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i+2][0])+" --beta "+ str(ABs[i+2][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i+2][0])+"b"+str(ABs[i+2][1])+".txt 2>&1 &"
#    command4 = "CUDA_VISIBLE_DEVICES=3 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i+3][0])+" --beta "+ str(ABs[i+3][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i+3][0])+"b"+str(ABs[i+3][1])+".txt 2>&1 "    
#    
#    print('command1', command1)
#    print('command2', command2)
#    print('command3', command3)
#    print('command4', command4)
#    os.system(command1)
#    os.system(command2)     
#    os.system(command3)
#    os.system(command4)

#for i in range(0, ABs.shape[0], 4 ):
#    print(i, ABs[i][0], ABs[i][1])
#    print(i +1, ABs[i+1][0], ABs[i+1][1])
#    print(i+2, ABs[i+2][0], ABs[i+2][1])
#    print(i+3, ABs[i+3][0], ABs[i+3][1])
      
## begin{additinal} 
#for i in range(0, 11, 2):
#    alpha = 0.1*1.2**(i-5)
#    beta = 1/alpha
#    if i == 0:
#        alphas_add = alpha
#        betas_add = beta
#    else:
#        alphas_add  = np.row_stack((alphas_add , alpha))
#        betas_add  = np.row_stack((betas_add , beta))
#    alpha2 = alpha*1.2
#    beta2 = 1/alpha2
#    alphas_add  = np.row_stack((alphas_add , alpha2))
#    betas_add  = np.row_stack((betas_add , beta2))
#  
#print('alphas_add , betas_add ', alphas_add , betas_add  )
#cnt = 0
#for i in alphas_add:
#    for j in betas_add:
#        
#        if cnt == 0:
#            ABs_more  = [i[0],j[0]]
#        else:
#            ABs_more  = np.row_stack((ABs_more, [ i[0],j[0] ]))
#        cnt = cnt + 1 
#
#print('ABs_more', ABs_more)
#
#cnt = 0 
#
#found = 0
#for AB in ABs_more:
#    for AB_temp in ABs:
#        if AB_temp[0] == AB[0] and AB_temp[1] == AB[1]:
#            found = 1
#    if found == 0:
#        if cnt == 0:
#            ABs_dif = AB
#        else:
#            ABs_dif  = np.row_stack((ABs_dif, AB))
#        cnt = cnt + 1 
#    found = 0    
#print('ABs_dif', ABs_dif)

#ABs_add 
#for i in range(0, ABs_dif.shape[0], 4 ):
#    print(i, ABs_dif[i][0], ABs_dif[i][1])
#    print(i+1, ABs_dif[i+1][0], ABs_dif[i+1][1])
#    print(i+2, ABs_dif[i+2][0], ABs_dif[i+2][1])
#    print(i+3, ABs_dif[i+3][0], ABs_dif[i+3][1])      
#      
#      
#    command1 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs_dif[i][0])+" --beta "+ str(ABs_dif[i][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs_dif[i][0])+"b"+str(ABs_dif[i][1])+".txt 2>&1 &"
#    command2 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs_dif[i+1][0])+" --beta "+ str(ABs_dif[i+1][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs_dif[i+1][0])+"b"+str(ABs_dif[i+1][1])+".txt 2>&1 &"
#    command3 = "CUDA_VISIBLE_DEVICES=2 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs_dif[i+2][0])+" --beta "+ str(ABs_dif[i+2][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs_dif[i+2][0])+"b"+str(ABs_dif[i+2][1])+".txt 2>&1 &"
#    command4 = "CUDA_VISIBLE_DEVICES=3 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs_dif[i+3][0])+" --beta "+ str(ABs_dif[i+3][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs_dif[i+3][0])+"b"+str(ABs_dif[i+3][1])+".txt 2>&1 "    
#    
#    print('command1', command1)
#    print('command2', command2)
#    print('command3', command3)
#    print('command4', command4)
#    os.system(command1)
#    os.system(command2)     
#    os.system(command3)
#    os.system(command4)


## end{additinal} 


#for i in range(0, ABs.shape[0], 4 ):
#    print(i, ABs[i][0], ABs[i][1])
#    print(i+1, ABs[i+1][0], ABs[i+1][1])
#    print(i+2, ABs[i+2][0], ABs[i+2][1])
#    print(i+3, ABs[i+3][0], ABs[i+3][1])      
#      
#      
#    command1 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i][0])+" --beta "+ str(ABs[i][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i][0])+"b"+str(ABs[i][1])+".txt 2>&1 &"
#    command2 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i+1][0])+" --beta "+ str(ABs[i+1][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i+1][0])+"b"+str(ABs[i+1][1])+".txt 2>&1 &"
#    command3 = "CUDA_VISIBLE_DEVICES=2 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i+2][0])+" --beta "+ str(ABs[i+2][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i+2][0])+"b"+str(ABs[i+2][1])+".txt 2>&1 &"
#    command4 = "CUDA_VISIBLE_DEVICES=3 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(ABs[i+3][0])+" --beta "+ str(ABs[i+3][1])+" > "+model_name+"_SGD_atanMomAtanA"+str(ABs[i+3][0])+"b"+str(ABs[i+3][1])+".txt 2>&1 "    
#    
##    print('command1', command1)
##    print('command2', command2)
##    print('command3', command3)
##    print('command4', command4)
##    os.system(command1)
##    os.system(command2)     
##    os.system(command3)
##    os.system(command4)
 
    #print (i)
#    command1 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha)+" --beta "+ str(beta)+" > "+model_name+"_SGD_atanMomAtanA"+str(alpha)+"b"+str(beta)+".txt 2>&1 &"
#    command2 = "CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch "+ model_name+" --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha "+str(alpha2)+" --beta "+ str(beta2)+" > "+model_name+"_SGD_atanMomAtanA"+str(alpha2)+"b"+str(beta2)+".txt 2>&1 "        
#print('alphas[0]*betas[-1]', alphas[0]*betas[-1])  
#print('alphas[-1]*betas[0]', alphas[-1]*betas[0])    
    
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch densenet_cifar --opt SGD_ori -bs 256 --lr 0.1 --workers 4  > densenet_cifar_ori.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch EfficientNetB0 --opt SGD_ori -bs 256 --lr 0.1 --workers 4  > EfficientNetB0_ori.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_ori -bs 256 --lr 0.1 --workers 4  > ResNet18_ori.txt 2>&1 ")

#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 2 --gamma 2 > ResNet18_Mom_Ada_Atan_shrink2Gamma2.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 2 --gamma 1.3 > ResNet18_Mom_Ada_Atan_shrink2Gamma1p3.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 2 --gamma 1.1 > ResNet18_Mom_Ada_Atan_shrink2Gamma1p1.txt 2>&1 ")
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 1.5 --gamma 2 > ResNet18_Mom_Ada_Atan_shrink1p5Gamma2.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 1.5 --gamma 1.5 > ResNet18_Mom_Ada_Atan_shrink1p5Gamma1p5.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 1.5 --gamma 1.3 > ResNet18_Mom_Ada_Atan_shrink1p5Gamma1p3.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 1.5 --gamma 1.1 > ResNet18_Mom_Ada_Atan_shrink1p5Gamma1p1.txt 2>&1 ")
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 1.3 --gamma 2 > ResNet18_Mom_Ada_Atan_shrink1p3Gamma2.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --shrink 1.3 --gamma 1.5 > ResNet18_Mom_Ada_Atan_shrink1p3Gamma1p5.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch LeNet --opt SGD_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > LeNet_SGD_ori.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 800 > LeNet_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > LeNet_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > LeNet_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > LeNet_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > LeNet_AtanA0p02b50.txt 2>&1 ") 
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > LeNet_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > LeNet_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > LeNet_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch LeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > LeNet_AtanA0p02b5.txt 2>&1 "  )




#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt Adam_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > ResNet18_Adam_ori.txt 2>&1 ") 
#
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt Adam -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > ResNet18_Adam_ori.txt 2>&1 ") 
#

#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch SigmoidLeNet -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 800 > SigmoidLeNet_AtanA0p02b800.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > ResNet18_SGD_atanMomAtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > ResNet18_SGD_atanMomAtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > ResNet18_SGD_atanMomAtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > ResNet18_SGD_atanMomAtanA0p02b50.txt 2>&1 ") 
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > ResNet18_SGD_atanMomAtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch ResNet18  --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > ResNet18_SGD_atanMomAtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch ResNet18  --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > ResNet18_SGD_atanMomAtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch ResNet18  --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > ResNet18_SGD_atanMomAtanA0p02b5.txt 2>&1 "  )
#
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet152 --opt SGD_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > LeNet_SGD_ori.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet152 --opt SGD_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > ResNet152_SGD_ori.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet18 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --alpha 999 --beta 666 > ResNet18_Mom_Ada_Atan.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet50 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --alpha 999 --beta 666 > ResNet50_Mom_Ada_Atan.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV1 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --alpha 999 --beta 666 > MobileV1Mom_Ada_Atan.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV2 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --alpha 999 --beta 666 > MobileV2_Mom_Ada_Atan.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --alpha 999 --beta 666 > EfficientNetB0_Mom_Ada_Atan.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch densenet_cifar --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --alpha 999 --beta 666 > densenet_cifar_Mom_Ada_Atan.txt 2>&1 ")
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom_Ada -bs 256 --lr 0.1 --workers 4 --alpha 999 --beta 666 > RegNetX_200MF_Mom_Ada_Atan.txt 2>&1 ")


#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > SigmoidLeNet_Mom_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > SigmoidLeNet_Mom_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > SigmoidLeNet_Mom_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > SigmoidLeNet_Mom_AtanA0p02b50.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > SigmoidLeNet_Mom_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > SigmoidLeNet_Mom_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > SigmoidLeNet_Mom_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch SigmoidLeNet --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > SigmoidLeNet_Mom_AtanA0p02b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet50 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > ResNet50_Mom_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet50 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > ResNet50_Mom_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > ResNet50_Mom_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > ResNet50_Mom_AtanA0p02b50.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > ResNet50_Mom_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > ResNet50_Mom_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > ResNet50_Mom_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch ResNet50 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > ResNet50_Mom_AtanA0p02b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV1 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > MobileV1_Mom_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV1 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > MobileV1_Mom_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > MobileV1_Mom_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > MobileV1_Mom_AtanA0p02b50.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > MobileV1_Mom_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > MobileV1_Mom_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > MobileV1_Mom_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch MobileV1 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > MobileV1_Mom_AtanA0p02b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV2 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > MobileV2_Mom_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV2 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > MobileV2_Mom_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > MobileV2_Mom_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > MobileV2_Mom_AtanA0p02b50.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > MobileV2_Mom_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > MobileV2_Mom_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > MobileV2_Mom_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch MobileV2 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > MobileV2_Mom_AtanA0p02b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > EfficientNetB0_Mom_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > EfficientNetB0_Mom_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > EfficientNetB0_Mom_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > EfficientNetB0_Mom_AtanA0p02b50.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > EfficientNetB0_Mom_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > EfficientNetB0_Mom_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > EfficientNetB0_Mom_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch EfficientNetB0 --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > EfficientNetB0_Mom_AtanA0p02b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch densenet_cifar --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > densenet_cifar_Mom_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch densenet_cifar --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > densenet_cifar_Mom_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > densenet_cifar_Mom_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > densenet_cifar_Mom_AtanA0p02b50.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > densenet_cifar_Mom_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > densenet_cifar_Mom_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > densenet_cifar_Mom_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch densenet_cifar --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > densenet_cifar_Mom_AtanA0p02b5.txt 2>&1 "  )
#
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 400 > RegNetX_200MF_Mom_AtanA0p02b400.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 200 > RegNetX_200MF_Mom_AtanA0p02b200.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 100 > RegNetX_200MF_Mom_AtanA0p02b100.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 50 > RegNetX_200MF_Mom_AtanA0p02b50.txt 2>&1 ") 
#os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 25 > RegNetX_200MF_Mom_AtanA0p02b25.txt 2>&1 ") 
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 10 > RegNetX_200MF_Mom_AtanA0p02b10.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 5 > RegNetX_200MF_Mom_AtanA0p02b5.txt 2>&1 " )
#os.system( "CUDA_VISIBLE_DEVICES=1 nohup python3  train_cifar10.py --arch RegNetX_200MF --opt SGD_atanMom  -bs 256 --lr 0.1 --workers 4 --alpha 0.02 --beta 2 > RegNetX_200MF_Mom_AtanA0p02b5.txt 2>&1 "  )






#CUDA_VISIBLE_DEVICES=2 nohup python3 train_RandLR.py --arch ResNet18 --LrSche RandLR --RandType uniform --opt SGD_ori -bs 256 --lr 0.15 --workers 4 --alpha -1 --beta -1 > ResNet18_RandLR_uniform_SGD_ex2.txt 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python3 train_RandLR.py --arch SigmoidLeNet --LrSche RandLR --RandType uniform --opt SGD_ori -bs 256 --lr 0.1 --workers 4 --alpha -1 --beta -1 > SigmoidLeNet_RandLR_uniform_SGD.txt 2>&1 &