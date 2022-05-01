from cProfile import label
import os.path
import torch
from torch import nn
import torch.optim as optim
from models.vae import VAE
from models.vae import VAEModel
from architectures import encoders, decoders
from common.ops import reparametrize
from common.utils import Accuracy_Loss, Interpretability
from common import constants as c
import torch.nn.functional as F
from common.utils import is_time_for

import numpy as np
import pandas as pd


class GrayVAE_Join(VAE):
    """
    Graybox version of VAE, with standard implementation. The discussion on
    """

    def __init__(self, args):

        super().__init__(args)

        print('Initialized GrayVAE_Join model')

        # checks
        assert self.num_classes is not None, 'please identify the number of classes for each label separated by comma'

        # encoder and decoder
        encoder_name = args.encoder[0]
        decoder_name = args.decoder[0]

        encoder = getattr(encoders, encoder_name)
        decoder = getattr(decoders, decoder_name)

        # number of channels
        image_channels = self.num_channels
        input_channels = image_channels
        decoder_input_channels = self.z_dim
        ## add binary classification layer
        if args.z_class is not None:
            self.z_class = args.z_class
        else:
            self.z_class = self.z_dim    
        self.n_classes = args.n_classes
        #self.classification = nn.Linear(self.z_dim, args.n_classes, bias=True).to(self.device) ### CHANGED OUT DIMENSION
        self.classification = nn.Linear(args.z_class, args.n_classes, bias=True).to(self.device) ### CHANGED OUT DIMENSION
        self.classification_epoch = args.classification_epoch
        self.reduce_rec = args.reduce_recon
        
        # model and optimizer
        self.model = VAEModel(encoder(self.z_dim, input_channels, self.image_size),
                               decoder(decoder_input_channels, self.num_channels, self.image_size),
                               ).to(self.device)
        self.optim_G = optim.Adam([*self.model.parameters(), *self.classification.parameters()],
                                      lr=self.lr_G, betas=(self.beta1, self.beta2))

        # update nets
        self.net_dict['G'] = self.model
        self.optim_dict['optim_G'] = self.optim_G

        self.setup_schedulers(args.lr_scheduler, args.lr_scheduler_args,
                              args.w_recon_scheduler, args.w_recon_scheduler_args)        

        ## CHOOSE THE WEIGHT FOR CLASSIFICATION
        self.label_weight = args.label_weight
        
        ## CHOOSE THE WEIGHT FOR LATENTS
        if args.latent_weight is None:
            self.latent_weight = args.label_weight 
        else:
            self.latent_weight = args.latent_weight
        self.masking_fact = args.masking_fact
        self.latent_loss = args.latent_loss
        
        ## OTHER STUFF
        self.show_loss = args.show_loss
        self.wait_counter = 0
        self.save_model = True

        self.dataframe_dis = pd.DataFrame() #columns=self.evaluation_metric)
        self.dataframe_eval = pd.DataFrame()
        self.validation_scores = pd.DataFrame()

    def predict(self, **kwargs):
        """
        Predict the correct class for the input data.
        """
        input_x = kwargs['latent'].to(self.device)
        pred_raw = self.classification(input_x)
        pred = nn.Softmax(dim=1)(pred_raw)
        return  pred_raw, pred.to(self.device, dtype=torch.float32) #nn.Sigmoid()(self.classification(input_x).resize(len(input_x)))

    def vae_classification(self, losses, x_true1, label1, y_true1, examples):

        mu, logvar = self.model.encode(x=x_true1,)

        z = reparametrize(mu, logvar)

        
        mu_processed = torch.tanh(z/2)
        
        #mu_processed = torch.tanh(mu/2)
 
        x_recon = self.model.decode(z=z,)

        #print('mu processed')
        #print(mu_processed[:10])

        prediction, forecast = self.predict(latent=mu_processed[:,:self.z_class])
        rn_mask = (examples==1)
        n_passed = len(examples[rn_mask])

        #print('Prediction:')
        #print(prediction[:10])

        
        loss_fn_args = dict(x_recon=x_recon, x_true=x_true1, mu=mu[label1.size(1):], logvar=logvar[label1.size(1):], z=z)
        loss_dict = self.loss_fn(losses, reduce_rec=False, **loss_fn_args)
        losses.update(loss_dict)

        pred_loss = nn.CrossEntropyLoss(reduction='mean')(prediction, y_true1) *self.label_weight  # her efor celebA
        losses.update(prediction=pred_loss)
        losses[c.TOTAL_VAE] += pred_loss

#            losses.update({'total_vae': loss_dict['total_vae'].detach(), 'recon': loss_dict['recon'].detach(),
#                          'kld': loss_dict['kld'].detach()})
        del loss_dict, pred_loss

        if n_passed > 0: # added the presence of only small labelled generative factors

            ## loss of categorical variables

            ## loss of continuous variables

            if self.latent_loss == 'MSE':
                #TODO: PLACE ONEHOT ENCODING
                
                loss_bin = nn.MSELoss(reduction='mean')( mu_processed[rn_mask][:, :label1.size(1)], 2*label1[rn_mask]-1  )
                ## track losses
                err_latent = []
                for i in range(label1.size(1)):
                    err_latent.append(nn.MSELoss(reduction='mean')(mu_processed[rn_mask][:, i], 2 * label1[rn_mask][:,i] - 1).detach().item() )
                losses.update(true_values=self.latent_weight * loss_bin)
                losses[c.TOTAL_VAE] += self.latent_weight * loss_bin

            elif self.latent_loss == 'BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, :label1.size(1)])/2,
                                                            label1[rn_mask] )

                ## track losses
                err_latent = []
                for i in range(label1.size(1)):
                    err_latent.append(nn.BCELoss(reduction='mean')((1+mu_processed[rn_mask][:, i])/2,
                                                                    label1[rn_mask][:, i] ).detach().item())
                losses.update(true_values=self.latent_weight * loss_bin)
                losses[c.TOTAL_VAE] += self.latent_weight * loss_bin

            else:
                raise NotImplementedError('Not implemented loss.')

        else:
            losses.update(true_values=torch.tensor(-1))
            err_latent =[-1]*label1.size(1)
    #            losses[c.TOTAL_VAE] += nn.MSELoss(reduction='mean')(mu[:, :label1.size(1)], label1).detach()


        return losses, {'x_recon': x_recon, 'mu': mu, 'z': z, 'logvar': logvar, "prediction": prediction,
                        'forecast': forecast, 'latents': err_latent, 'n_passed': n_passed}

    def train(self, **kwargs):

        if 'output'  in kwargs.keys():
            out_path = kwargs['output']
            track_changes=True
            self.out_path = out_path #TODO: Not happy with this thing
            print("## Initializing Train indexes")
            print("->path chosen::",out_path)

        else: track_changes=False;
            
        
        ## SAVE INITIALIZATION ##
        #self.save_checkpoint()


        Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores = [], [], [], [], [], [], []  ## JUST HERE FOR NOW
        latent_errors = []
        epoch = 0
        self.optim_G.param_groups[0]['lr'] = 0
        lr_log_scale = np.logspace(-7,-4, 20)
        while not self.training_complete():
            # added annealing
            if epoch < 10:
                self.optim_G.param_groups[0]['lr'] = lr_log_scale[epoch] 
            print('lr:',  self.optim_G.param_groups[0]['lr'])
            epoch += 1
            self.net_mode(train=True)
            vae_loss_sum = 0
            # add the classification layer #
            if epoch>self.classification_epoch:
                print("## STARTING CLASSIFICATION ##")
                start_classification = True
            else: start_classification = False
            
            # to evaluate performances on disentanglement
            z = torch.zeros(self.batch_size*len(self.data_loader), self.z_dim, device=self.device)
            g = torch.zeros(self.batch_size*len(self.data_loader), self.z_dim, device=self.device)
            
            for internal_iter, (x_true1, label1, y_true1, examples) in enumerate(self.data_loader):

                losses = {'total_vae':0}

                x_true1 = x_true1.to(self.device)
                y_true1 = y_true1.to(self.device, dtype=torch.long)

                #label1 = label1[:, 1:].to(self.device) #TODO CHANGE THE 1 with SOMETHING CHOSEN
                if self.dset_name == 'dsprites_full':
                    label1 = label1[:, 1:].to(self.device)
                else:
                    label1 = label1.to(self.device)

                
                
                losses, params = self.vae_classification(losses, x_true1, label1, y_true1, examples)

                ## ADD FOR EVALUATION PURPOSES
                z[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size, :] = params['z']
                g[internal_iter*self.batch_size:(internal_iter+1)*self.batch_size, :label1.size(1)] = label1

                self.optim_G.zero_grad()

                if (internal_iter%self.show_loss)==0: print("Losses:", losses)

                if not start_classification:
                    losses[c.TOTAL_VAE].backward(retain_graph=False)
                    #losses['true_values'].backward(retain_graph=False)
                    self.optim_G.step()

                if start_classification:   # and (params['n_passed']>0):
                    losses['prediction'].backward(retain_graph=False)
                    self.optim_G.step()

                vae_loss_sum += losses[c.TOTAL_VAE]
                losses[c.TOTAL_VAE_EPOCH] = vae_loss_sum /( internal_iter+1) ## ADDED +1 HERE IDK WHY NOT BEFORE!!!!!

                ## Insert losses -- only in training set
                if track_changes and is_time_for(self.iter, self.test_iter):
                    #TODO: set the tracking at a given iter_number/epoch
                    print('tracking changes')
                    Iterations.append(self.iter + 1); Epochs.append(epoch)
                    Reconstructions.append(losses['recon'].item()); KLDs.append(losses['kld'].item()); True_Values.append(losses['true_values'].item())
                    latent_errors.append(params['latents']); Accuracies.append(losses['prediction'].item())
                    F1_scores.append(Accuracy_Loss()(params['prediction'], y_true1, dims=self.n_classes).item())
                    
                    if epoch >0:
                        sofar = pd.DataFrame(data=np.array([Iterations, Epochs, Reconstructions, KLDs, True_Values, Accuracies, F1_scores]).T,
                                             columns=['iter', 'epoch', 'reconstruction_error', 'kld', 'latent_error', 'classification_error', 'accuracy'], )
                        for i in range(label1.size(1)):
                            sofar['latent%i'%i] = np.asarray(latent_errors)[:,i]

                        sofar.to_csv(os.path.join(out_path+'/train_runs', 'metrics.csv'), index=False)
                        del sofar

#                        if not self.dataframe_eval.empty:
 #                           self.dataframe_eval.to_csv(os.path.join(out_path, 'dis_metrics.csv'), index=False)
                        # ADD validation step
                        val_rec, val_kld, val_latent, val_bce, val_acc, _, _, _ =self.test(validation=True, name=self.dset_name)
                        sofar = pd.DataFrame(np.array([epoch, val_rec, val_kld, val_latent, val_bce, val_acc]).reshape(1,-1), 
                                            columns=['epoch','rec', 'kld', 'latent', 'bce', 'acc'] )
                        self.validation_scores = self.validation_scores.append(sofar, ignore_index=True)
                        self.validation_scores.to_csv(os.path.join(out_path+'/train_runs', 'val_metrics.csv'), index=False)
                        del sofar
                    # validation check
                    if epoch > 20: 
                        print('Validation stop evaluation')
                        print(self.iter, self.epoch)
                        print(self.validation_scores)
                        self.validation_stopping()


                # TESTSET LOSSES
                if is_time_for(self.iter, self.test_iter):

                    #                    self.dataframe_eval = self.dataframe_eval.append(self.evaluate_results,  ignore_index=True)
                    # test the behaviour on other losses
                    trec, tkld, tlat, tbce, tacc, I, I_tot, err_latent = self.test(end_of_epoch=False,name=self.dset_name, 
                                                                        out_path=self.out_path )
                    factors = pd.DataFrame(
                        {'iter': self.iter+1, 'rec': trec, 'kld': tkld, 'latent': tlat, 'BCE': tbce, 'Acc': tacc,
                         'I': I_tot}, index=[0])

                    for i in range(len(err_latent)):
                        factors['latent%i' % i] = np.asarray(err_latent)[i]

                    self.dataframe_eval = self.dataframe_eval.append(factors, ignore_index=True)
                    self.net_mode(train=True)

                    if track_changes and not self.dataframe_eval.empty:
                        self.dataframe_eval.to_csv(os.path.join(out_path, 'eval_results/test_metrics.csv'),
                                                   index=False)

                    # include disentanglement metrics
                    dis_metrics = pd.DataFrame(self.evaluate_results, index=[0])
                    self.dataframe_dis = self.dataframe_dis.append(dis_metrics)
                    del dis_metrics

                    if track_changes and not self.dataframe_dis.empty:
                        self.dataframe_dis.to_csv(os.path.join(out_path, 'eval_results/dis_metrics.csv'),
                                                  index=False)
                        print('Saved dis_metrics')

                    
                if self.save_model:
                    self.log_save(input_image=x_true1, recon_image=params['x_recon'], loss=losses)
                else:
                    self.step()
                    pass_dict ={'input_image':x_true1, 'recon_image':params['x_recon'], 'loss':losses}
                    if is_time_for(self.iter, self.schedulers_iter):
                        self.schedulers_step(pass_dict.get(c.LOSS, dict()).get(c.TOTAL_VAE_EPOCH, 0),
                                            self.iter // self.schedulers_iter)
                    del pass_dict

            # end of epoch
            if self.save_model:
                print('Saved model at epoch', self.epoch)
            
            if out_path is not None and self.save_model: # and validation is None:
                with open( os.path.join(out_path,'train_runs/latents_obtained.npy'), 'wb') as f:
                    np.save(f, z.detach().cpu().numpy())
                    np.save(f, g.detach().cpu().numpy())
                del z, g
                
        self.pbar.close()

    def test(self, end_of_epoch=True, validation=False, name='dsprites_full', out_path=None):
        self.net_mode(train=False)
        rec, kld, latent, BCE, Acc = 0, 0, 0, 0, 0
        I = np.zeros(self.z_dim)
        I_tot = 0

        N = 10**4
        l_dim = self.z_dim
        g_dim = self.z_dim

        z_array = np.zeros( shape=(self.batch_size*len(self.test_loader), l_dim))
        g_array = np.zeros( shape=(self.batch_size*len(self.test_loader), g_dim))

        if validation: loader = self.val_loader
        else: loader = self.test_loader

        for internal_iter, (x_true, label, y_true, _) in enumerate(loader):
            x_true = x_true.to(self.device)

            if self.dset_name == 'dsprites_full':
                label = label[:, 1:].to(self.device)
            else:
                label = label.to(self.device)
            
            y_true =  y_true.to(self.device, dtype=torch.long)

            g_array = g_array[:,:label.size(1)]

            mu, logvar = self.model.encode(x=x_true, )
            z = reparametrize(mu, logvar)

            mu_processed = torch.tanh(z / 2)
            prediction, forecast = self.predict(latent=mu_processed[:,:self.z_class])
            x_recon = self.model.decode(z=z,)

            z = np.asarray(nn.Sigmoid()(z).detach().cpu())
            g = np.asarray(label.detach().cpu())

            z_array[self.batch_size*internal_iter:self.batch_size*internal_iter+self.batch_size, :] = z
            g_array[self.batch_size*internal_iter:self.batch_size*internal_iter+self.batch_size, :] = g

#            I_batch , I_TOT = Interpretability(z, g)
 #           I += I_batch; I_tot += I_TOT

            rec+=(F.binary_cross_entropy(input=x_recon, target=x_true,reduction='sum').detach().item()/self.batch_size )
            kld+=(self._kld_loss_fn(mu, logvar).detach().item())

            if self.latent_loss == 'MSE':
                loss_bin = nn.MSELoss(reduction='mean')(mu_processed[:, :label.size(1)], 2 * label.to(dtype=torch.float32) - 1)
                
                err_latent = []
                for i in range(label.size(1)):
                    err_latent.append(nn.MSELoss(reduction='mean')(mu_processed[:, i],
                                                                    2*label[:, i].to(dtype=torch.float32)-1 ).detach().item())
            elif self.latent_loss == 'BCE':
                loss_bin = nn.BCELoss(reduction='mean')((1+mu_processed[:, :label.size(1)])/2, label.to(dtype=torch.float32) )
                
                err_latent = []
                for i in range(label.size(1)):
                    err_latent.append(nn.BCELoss(reduction='mean')((1+mu_processed[:, i])/2,
                                                                    label[:, i] ).detach().item())
            else:
                NotImplementedError('Wrong argument for latent loss.')

            latent+=(loss_bin.detach().item())
            del loss_bin

            BCE+=(nn.CrossEntropyLoss(reduction='mean')(prediction,
                                                        y_true).detach().item())


            Acc+=(Accuracy_Loss()(forecast,
                                y_true, dims=self.n_classes).detach().item() )

        if end_of_epoch:
            self.visualize_recon(x_true, x_recon, test=True)
            self.visualize_traverse(limit=(self.traverse_min, self.traverse_max),
                                    spacing=self.traverse_spacing,
                                    data=(x_true, label), test=True)

            #self.iter += 1
            #self.pbar.update(1)
        if out_path is not None and self.save_model: # and validation is None:
            with open( os.path.join(out_path,'eval_results/latents_obtained.npy'), 'wb') as f:
                np.save(f, z_array)
                np.save(f, g_array)
        
            
        nrm = internal_iter + 1
        return rec/nrm, kld/nrm, latent/nrm, BCE/nrm, Acc/nrm, I/nrm, I_tot/nrm, [err/nrm for err in err_latent]

    