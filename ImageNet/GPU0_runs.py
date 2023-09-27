import os

os.system("CUDA_VISIBLE_DEVICES=9,8,7,6,3,2,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210846 main_cosine.py --arch resnet18 --opt SGD_atanMom -bs 256 --lr 0.4 --opt_level O2 --workers 8 --alpha 0.2 --beta 10  > res18_SGD_atanMomAtan_a0p2b10.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,9,8,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210847 main_cosine.py --arch resnet18 --opt SGD_atanMom -bs 256 --lr 0.4 --opt_level O2 --workers 8 --alpha 0.1 --beta 20  > res18_SGD_atanMomAtan_a0p1b20.txt 2>&1")
os.system(" CUDA_VISIBLE_DEVICES=3,2,1,0,9,8,7,6 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210845 main_cosine.py --arch resnet18 --opt SGD_ori -bs 256 --lr 0.4 --opt_level O2 --workers 8 --alpha -1 --beta -1 > res18_SGD_ori.txt 2>&1")

os.system("CUDA_VISIBLE_DEVICES=3,2,1,0,9,8,7,6 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210848 main_cosine.py --arch resnet34 --opt SGD_ori -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha -1 --beta -1  > res34_SGD_ori.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=9,8,7,6,3,2,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210849 main_cosine.py --arch resnet34 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.2 --beta 10  > res34_SGD_atanMomAtan_a0p2b10.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,9,8,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210850 main_cosine.py --arch resnet34 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.1 --beta 20  > res34_SGD_atanMomAtan_a0p1b20.txt 2>&1")


os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,4,5,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 2238 main_cosine.py --arch resnet50 --clip norm --opt SGD_ori -bs 64 --lr 0.1 --opt_level O2 --workers 8 --alpha 0.1 --beta -1  > res50_SGD_clip_norm_0.1.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,4,5,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 2239 main_cosine.py --arch resnet50 --clip value --opt SGD_ori -bs 64 --lr 0.1 --opt_level O2 --workers 8 --alpha 0.1 --beta -1  > res50_SGD_clip_value_0.1.txt 2>&1")

os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,4,5,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 2240 main_cosine.py --arch se_resnet50 --clip norm --opt SGD_ori -bs 64 --lr 0.1 --opt_level O2 --workers 8 --alpha 0.1 --beta -1  > se_res50_SGD_clip_norm_0.1.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,4,5,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 2241 main_cosine.py --arch se_resnet50 --clip value --opt SGD_ori -bs 64 --lr 0.1 --opt_level O2 --workers 8 --alpha 0.1 --beta -1  > se_res50_SGD_clip_value_0.1.txt 2>&1")



os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,9,8,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 211107 main_cosine.py --arch resnet18 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.1 --beta 20  > res18_SGD_atanMomAtan_a0p1b20.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=9,8,7,6,3,2,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 211106 main_cosine.py --arch se_resnet18 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 9999 --beta 9999  > se_res18_SGD_ori.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,9,8,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 211108 main_cosine.py --arch se_resnet18 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.1 --beta 20  > se_res18_SGD_atanMomAtan_a0p1b20.txt 2>&1")
os.system("CUDA_VISIBLE_DEVICES=7,6,3,2,9,8,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 211109 main_cosine.py --arch se_resnet18 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.2 --beta 10  > se_res18_SGD_atanMomAtan_a0p2b10.txt 2>&1")




#os.system(" CUDA_VISIBLE_DEVICES=3,2,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 182124 main_cosine.py --arch resnet18 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.1 --beta 20 > res18_SGD_atanMomAtan_a0p1b20.txt 2>&1")
#os.system(" CUDA_VISIBLE_DEVICES=9,8,7,6,3,2,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 182126 main_cosine.py --arch resnet50 --opt SGD_atanMom -bs 64 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.2 --beta 10 > res50_SGD_atanMomAtan_a0p1b20.txt 2>&1")

#os.system(" CUDA_VISIBLE_DEVICES=3,2,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 182124 main_cosine.py --arch resnet18 --opt SGD_atanMom -bs 256 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.1 --beta 20 > res18_SGD_atanMomAtan_a0p1b20.txt 2>&1")
#os.system(" CUDA_VISIBLE_DEVICES=9,8,7,6,3,2,1,0 nohup python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 182126 main_cosine.py --arch resnet50 --opt SGD_atanMom -bs 128 --lr 0.2 --opt_level O2 --workers 8 --alpha 0.2 --beta 10 > res50_SGD_atanMomAtan_a0p1b20.txt 2>&1")
#CUDA_VISIBLE_DEVICES=3,2,1,0,9,8,7,6 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210837 main_cosine.py --arch resnet18 --opt SGD_atanMom -bs 256 --lr 0.4 --opt_level O2 --workers 8 --alpha 0.2 --beta 10 --notes 8GPUs > res18_SGD_atanMomAtan_a0p2b10.txt 2>&1

#CUDA_VISIBLE_DEVICES=3,2,1,0,9,8,7,6 nohup python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 210837 main_cosine.py --arch resnet18 --opt SGD_ori -bs 256 --lr 0.4 --opt_level O2 --workers 8 --alpha 0.2 --beta 10 > res18_SGD_ori_a0p2b10.txt 2>&1 &
