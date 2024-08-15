CUDA_VISIBLE_DEVICES=0 python main.py --scheduler=one --kernel=t --alpha=2.0 --beta=0.15 --gamma=0.75 --model_name FarconVAE-t

CUDA_VISIBLE_DEVICES=0 python main.py --scheduler=one  --kernel=g --alpha=1.0 --beta=0.05 --gamma=1.0 --model_name FarconVAE-G

CUDA_VISIBLE_DEVICES=0 python main_maxent.py --scheduler=one --alpha=4.0 --model_name MaxEnt-ARL

CUDA_VISIBLE_DEVICES=0 python main_maxent.py --scheduler=one --alpha=4.0 --model_name CI

CUDA_VISIBLE_DEVICES=0 python main_uf.py 

CUDA_VISIBLE_DEVICES=0 python main.py 

python evaluate.py