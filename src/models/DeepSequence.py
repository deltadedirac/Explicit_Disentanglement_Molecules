from .encoder_decoder import mlp_encoder,mlp_decoder
import torch.distributions as D

from tqdm import tqdm

import torch
import numpy as np
import time, os, datetime, gc, pdb, pickle, copy


class DeepSequence(torch.nn.Module):

    def __init__(self, 
                 input_shape, 
                 latent_dim, 
                 alphabet, 
                 device = 'cpu', 
                 outputnonlin=torch.nn.Softmax(dim=-1),
                 beta=1):
        super(DeepSequence, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.alphabet = alphabet
        self.outputnonlin = outputnonlin

        self.beta = beta

        self.device = (device,'cuda')[device =='gpu' or device =='cuda']

        # Define encoder and decoder
        self.encoder = mlp_encoder(input_shape, latent_dim, 
                                   layer_ini = self.alphabet, dropout=0.0).to(self.device)
        self.decoder = mlp_decoder(input_shape, latent_dim, self.outputnonlin, 
                                   layer_ini = self.alphabet, dropout=0.0).to(self.device)


    def KL(self, q_dist, prior):

        #import ipdb; ipdb.set_trace()
        kl = D.kl_divergence(q_dist, prior)
        return kl
    

    def reparameterize(self, q_dist):
        #import ipdb; ipdb.set_trace()  
        return q_dist.rsample()
        

    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0):

        #import ipdb; ipdb.set_trace()
        # Encode/decode semantic space
        mu, var = self.encoder(x)
        
        # Create a variational distribution from the
        # calculated mu and var in the encoder layer
        q_dist = D.Independent( 
            D.Normal(mu, torch.exp(0.5 * var)), 1)

        
        # Define the prior as isotropic gaussian
        self.prior = D.Independent( 
                            D.Normal(
                            torch.zeros_like(q_dist.mean).to(self.device), 
                            torch.ones_like(q_dist.variance).to(self.device)
                            ), 1 )

        z = self.reparameterize(q_dist)
        
        KLD = self.KL(q_dist, self.prior) 
        x_mean, x_var = self.decoder(z)  

        x_var = switch*x_var + (1-switch)*0.02**2
        
        return x_mean.contiguous(), \
                x_var.contiguous(), \
                z, mu, var, KLD

    def sample(self, n, switch=1.0):
        device = self.device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            x_mean, x_var = self.decoder(z)
            x_var = switch*x_var + (1-switch)*0.02**2
            return x_mean.contiguous(), \
                            x_var.contiguous(), \
                                z

    def training_representation( self, trainloader, 
                                 optimizer, 
                                 n_epochs=10, 
                                 warmup=1, 
                                 logdir='', 
                                 out_modelname='trained_model.pth',
                                 eq_samples=1, 
                                 iw_samples=1,
                                 save_best_checkpoint = False):

        # Main loop
        
        best_fit = 1e6; best_epoch = 0
        best_model = self

        self.train()
        start = time.time()
        #import ipdb; ipdb.set_trace()
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch) + '/' + str(n_epochs), 
                                total=len(trainloader.dataset), unit='samples')
            # Training loop
            for i, data in enumerate(trainloader):
                # Zero gradient
                optimizer.zero_grad()

                # Feed forward data
                data = data.to(torch.float32).to(self.device)

                switch = 1.0 if epoch > warmup else 0.0
                out = self.forward(data, eq_samples, iw_samples, switch)
                

                loss = self.get_elbo(data, out, reduction='none')
                loss_scalar = loss.mean()
                # Backpropegate and optimize
                loss_scalar.backward()
                optimizer.step()
                
                # Write to consol
                progress_bar.update(data.size(0))
                progress_bar.set_postfix({'loss': loss_scalar.item()})


                if loss_scalar < best_fit:
                    best_fit = loss_scalar
                    best_model.load_state_dict( copy.deepcopy( self.state_dict() ) )
                    best_epoch = epoch

                gc.collect()
                torch.cuda.empty_cache()
            progress_bar.close()
        
        print('Total train time', time.time() - start)
        #print(f'best fit: {best_fit} \n best epoch: {best_epoch} \n\n Model: {best_model}')
        #self.load_state_dict( copy.deepcopy( best_model.state_dict() ) )

        torch.save(self.state_dict(), logdir + '/' + out_modelname)



    def get_elbo(self, data, out, reduction='none'):
 
        ELBO = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                                                data.argmax(-1), reduction = "none")
                                                + self.beta*(out[5]).reshape(-1,1)
                                                #+ self.beta*(out[5]).mean(-1).reshape(-1,1) 
                                                ).mean(-1)
        
        if reduction == 'mean':
            return ELBO.mean()
        else:
            return ELBO
