python main.py \
--name=$NAME \
--alg=CBM_Join \
--dset_dir=data \
--dset_name=mpi3d_toy \
--d_version=full \
--gif_save=False \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=21 \
--z_class=21 \
--batch_size=100 \
--use_wandb=false \
--test_iter=2500 \
--evaluate_iter=1000000 \
--classification_epoch=1000 \
--max_epoch=15 \
--latent_loss=BCE \
--n_classes=10 \
--seed=883 \
--label_weight=1 \
--latent_weight=2 \
--lr_scheduler=ExponentialLR \
--lr_G=0.001 \
--lr_scheduler_args=gamma=0.75 \
--masking_fact=1 \
