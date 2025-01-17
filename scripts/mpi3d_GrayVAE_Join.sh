python main.py \
--name=$NAME \
--alg=GrayVAE_Join \
--dset_dir=data  \
--dset_name=mpi3d_toy \
--gif_save=False \
--d_version=full \
--encoder=PadlessGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=21 \
--z_class=21 \
--batch_size=100 \
--use_wandb=false \
--test_iter=8294 \
--evaluate_iter=1000000 \
--classification_epoch=1000 \
--max_epoch=30 \
--latent_loss=BCE \
--n_classes=10 \
--seed=883 \
--w_recon=1 \
--max_c=40 \
--iterations_c=100000 \
--label_weight=1000 \
--latent_weight=7000 \
--lr_G=0.001 \
--lr_scheduler=ExponentialLR \
--lr_scheduler_args=gamma=0.99 \
--w_kld=1 \
--masking_fact=1 \
