This repository is used to reproduce the results in paper: ```On the Connection Between Counterfactual Fairness, Statistical Parity and Individual Fairness```.

Directory ```GCM_Simulation``` contains the code for the experiment with synthetic data generated by GCM. To get the result in Table 1, run the following command:
```sh
cd GCM_Simulation
python main.py
```

The following directory contains codes developed based on the repository of FarconVAE ([https://github.com/changdaeoh/FarconVAE](https://github.com/changdaeoh/FarconVAE)).

Directory ```Non_GCM_Simulation``` contains the code for the experiment with synthetic data generated beyond GCM. To get the result in Table 2, run the following command:
```sh
# generate representations
CUDA_VISIBLE_DEVICES=0 python main.py --scheduler=one --kernel=t --alpha=2.0 --beta=0.15 --gamma=0.75 --model_name FarconVAE-t

CUDA_VISIBLE_DEVICES=0 python main.py --scheduler=one  --kernel=g --alpha=1.0 --beta=0.05 --gamma=1.0 --model_name FarconVAE-G

CUDA_VISIBLE_DEVICES=0 python main_maxent.py --scheduler=one --alpha=4.0 --model_name MaxEnt-ARL

CUDA_VISIBLE_DEVICES=0 python main_maxent.py --scheduler=one --alpha=4.0 --model_name CI

python main_uf.py

python main_cf.py

# evaluate representations
python evaluate.py
```

