# Continually trained classifier

#### Split-MNIST
```
python main.py --experiment_name MNIST_example --dataset MNIST --gpuid 0 --num_batches 5 --gen_d=32 --gen_latent_size 8 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --skip_normalization --seed 13 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --gen_batch_size 64 --training_procedure classifier --fe_type mlp400 --generator_type vae --skip_validation
```

#### MNIST Dirichlet split
```
 python main.py --experiment_name MNIST_example_dirichlet --dataset MNIST --gpuid 0 --num_batches 10 --gen_d=32 --gen_latent_size 8 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --skip_normalization --seed 13 --score_on_val --cosine_sim 0.95 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --dirichlet 1 --gen_batch_size 64 --training_procedure classifier  --fe_type mlp400 --generator_type vae --skip_validation
```

#### Split-FashionMNIST

```
python main.py --experiment_name FashionMNIST_example --dataset FashionMNIST --gpuid 0 --num_batches 5 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.98 --global_lr 0.001 --local_lr 0.0005 --binary_latent_size 4 --global_warmup 5 --local_scheduler_rate 0.98 --scale_reconstruction_loss 5 --global_scheduler_rate 0.98 --gen_batch_size 64 --training_procedure classifier --fe_type resnet18 --generator_type vae --skip_validation 
```

#### FashionMNIST Dirichlet split
```
python main.py --experiment_name FashionMNIST_example_dirichlet --dataset FashionMNIST --gpuid 0 --num_batches 10 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.98 --global_lr 0.001 --local_lr 0.0005 --binary_latent_size 4 --global_warmup 5 --local_scheduler_rate 0.98 --scale_reconstruction_loss 5 --dirichlet 1 --global_scheduler_rate 0.98 --gen_batch_size 64 --training_procedure classifier --fe_type mlp400 --generator_type vae --skip_validation
```

#### DoubleMNIST
```
python main.py --experiment_name DoubleMNIST_example --dataset DoubleMNIST --gpuid 0 --num_batches 10 --gen_batch_size 64 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --global_warmup 5 --local_scheduler_rate 0.99 --scale_reconstruction_loss 3 --training_procedure classifier --fe_type mlp400 --generator_type vae --skip_validation
```

#### Omniglot
```
python main.py --experiment_name Omniglot_example --dataset Omniglot --gpuid 0 --num_batches 20 --gen_batch_size 64 --gen_d=32 --gen_latent_size 16 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 6 --gen_p_coding 13 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 12 --global_warmup 5 --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --scale_reconstruction_loss 10 --training_procedure classifier
```

---- updated commands ----

#### CI5-MNIST
```
python main.py --experiment_name MNIST_example --dataset MNIST --num_classes 10 --gpuid 0  --seed 13 --optimizer Adam --mse_reduction  --batch_size 64 --fe_type mlp400 --feature_extractor_epochs 30 --generator_type gan --final_task_only --calc_noise --log_wandb
```

#### CI5-CIFAR10
```
python main.py --experiment_name CIFAR10_example --dataset CIFAR10 --num_classes 10 --gpuid 0 --seed 13 --optimizer Adam --mse_reduction --batch_size 64 --fe_type resnet18 --feature_extractor_epochs 30 --generator_type gan --calc_noise --train_aug --cutmix --log_wandb
```

#### CI20-CIFAR100
```
python main.py --experiment_name CIFAR100_example --dataset CIFAR100 --num_classes 100 --gpuid 0 --seed 13 --optimizer Adam --mse_reduction --batch_size 256 --fe_type resnet34 --feature_extractor_epochs 30 --generator_type gan --calc_noise --train_aug --cutmix --log_wandb
```

