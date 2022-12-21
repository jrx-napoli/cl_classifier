# cl_classifier

#### Split-MNIST
```
python main.py --experiment_name MNIST_example --dataset MNIST --gpuid 0 --num_batches 5 --gen_d=32 --gen_latent_size 8 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --skip_normalization --seed 13 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --fc --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --gen_batch_size 32 --training_procedure classifier
```

```
python main.py --experiment_name MNIST_example --dataset MNIST --gpuid 0 --num_batches 5 --gen_d=32 --gen_latent_size 8 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --skip_normalization --seed 13 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --gen_batch_size 32 --training_procedure classifier
```

#### MNIST Dirichlet split
```
 python main.py --experiment_name MNIST_example_dirichlet --dataset MNIST --gpuid 0 --num_batches 10 --gen_d=32 --gen_latent_size 8 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --skip_normalization --seed 13 --score_on_val --cosine_sim 0.95 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --fc --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --dirichlet 1 --gen_batch_size 32 --training_procedure classifier
```

#### Split-FashionMNIST
```
python main.py --experiment_name FashionMNIST_example --dataset FashionMNIST --gpuid 0 --num_batches 5 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.98 --global_lr 0.001 --local_lr 0.0005 --binary_latent_size 4 --global_warmup 5 --fc --local_scheduler_rate 0.98 --scale_reconstruction_loss 5 --global_scheduler_rate 0.98 --gen_batch_size 32 --training_procedure classifier
```

```
python main.py --experiment_name FashionMNIST_example --dataset FashionMNIST --gpuid 0 --num_batches 5 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.98 --global_lr 0.001 --local_lr 0.0005 --binary_latent_size 4 --global_warmup 5 --local_scheduler_rate 0.98 --scale_reconstruction_loss 5 --global_scheduler_rate 0.98 --gen_batch_size 32 --training_procedure classifier --skip_validation
```

#### FashionMNIST Dirichlet split
```
python main.py --experiment_name FashionMNIST_example_dirichlet --dataset FashionMNIST --gpuid 0 --num_batches 10 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.98 --global_lr 0.001 --local_lr 0.0005 --binary_latent_size 4 --global_warmup 5 --fc --local_scheduler_rate 0.98 --scale_reconstruction_loss 5 --dirichlet 1 --global_scheduler_rate 0.98 --gen_batch_size 32 --training_procedure classifier
```







python main.py --experiment_name CIFAR10_example --dataset CIFAR10 --gpuid 1 --num_batches 5 --gen_batch_size 64 --gen_d=50 --gen_latent_size 32 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.95 --global_lr 0.003 --local_lr 0.001 --binary_latent_size 8 --global_warmup 5