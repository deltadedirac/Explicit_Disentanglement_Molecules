#%%
import torch
from torch import nn
from torchvision.utils import make_grid
import numpy as np
import pdb, gc

from ..unsuper.unsuper.helper.utility import affine_decompose, Identity
from .spatial_transformers_modes import get_transformer
PI = torch.from_numpy(np.asarray(np.pi))


#%%
class VITAE_CI(nn.Module):
    def __init__(self, input_shape, config, latent_dim, encoder, decoder, outputdensity, ST_type, **kwargs):
        super(VITAE_CI, self).__init__()
        # Constants

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = latent_dim
        self.outputdensity = outputdensity
        self.alphabet = kwargs["alphabet_size"]

        # list of indexes for each sequences for padded 
        # and nonpadded parts of batch sequences to process
        '''
        self.padded_idx_seqs = kwargs["padded_idx"]
        self.non_padded_idx_seqs =  kwargs["non_padded_idx"]
        '''
        pdb.set_trace()
        # Spatial transformer
        self.stn = get_transformer(ST_type)(#input_shape,
                                            [ config.parserinfo('*/Window_grid') ], 
                                            config)
        self.ST_type = ST_type

        self.Trainprocess = True
        self.x_trans1 = torch.tensor([])
        # Define outputdensities
        if outputdensity == 'bernoulli':
            #outputnonlin = nn.Sigmoid()
            outputnonlin = nn.Softmax(dim=1)
        elif outputdensity == 'gaussian':
            outputnonlin = Identity()
        elif outputdensity == 'softmax':
            outputnonlin = nn.Softmax(dim=2)
        elif outputdensity == 'log_softmax':
            outputnonlin = nn.LogSoftmax(dim=2)
        else:
            ValueError('Unknown output density')
        
        # Define encoder and decoder
        pdb.set_trace()
        if isinstance(encoder,tuple) and isinstance(decoder,tuple):
            self.encoder1 = encoder[0](input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder1 = decoder[0]((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet)
            
            self.encoder2 = encoder[1](input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder2 = decoder[1](input_shape, latent_dim, outputnonlin, layer_ini = self.alphabet)
        else:
            self.encoder1 = encoder(input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder1 = decoder((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet)
            
            self.encoder2 = encoder(input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder2 = decoder(input_shape, latent_dim, outputnonlin, layer_ini = self.alphabet)
        # CHECK CAREFULLY!!!!!!!!!!! INCLUDING THE OUTPUT FROM GP_CPAB
        #self.dim_transformation = torch.nn.parameter.Parameter(data= torch.tensor(self.input_shape).float() ) #, requires_grad=True)
    '''
    #%%
    def train(self):
        self.Trainprocess = True

    #%%
    def eval(self):
        self.Trainprocess =  False
    '''
    def log_normal_diag(self, x, mu, log_var, reduction=None, dim=None):
        D = x.shape[1]
        log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p


    def log_standard_normal(self, x, reduction=None, dim=None):
        D = x.shape[1]
        log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p

    def KL(self, z, mu, log_var):
        log_z = self.log_standard_normal(z)
        log_qz = self.log_normal_diag(z, mu, log_var)
        return ( log_z - log_qz ).mean()
    #%%
    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:].sqrt() * eps).reshape(-1, latent_dim)
    
    #%%
    
    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0,trainprocess=False):
        # Encode/decode transformer space
        #pdb.set_trace()
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)
        KLD1 = self.KL(z1, mu1, var1)

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), theta_mean, self.Trainprocess, inverse=True)
        self.x_trans1 = x_new
        
        '''-------------------------------------------------------------------------------------------------------'''
        
        # Encode/decode semantic space
        mu2, var2 = self.encoder2(x_new)
        z2 = self.reparameterize(mu2, var2, 1, 1)
        x_mean, x_var = self.decoder2(z2)
        KLD2 = self.KL(z2, mu2, var2)

        #self.xmean_before_reverse = x_mean 
        # "Detransform" output
        
        '''-------------------------------------------------------------------------------------------------------'''

        x_mean = x_new #; z2=mu2=var2=None; x_var=torch.ones(1,3,3)
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_mean = self.stn(x_mean, theta_mean,  self.Trainprocess, inverse=False)
        '''x_var = self.stn(x_var, theta_mean, self.Trainprocess, inverse=False)


        x_var = switch*x_var + (1-switch)*0.02**2'''
        
        return x_mean.unsqueeze(1).contiguous(), \
                x_var.unsqueeze(1).contiguous(), [z1, z2], [mu1, mu2], [var1, var2], x_new, theta_mean, KLD1, KLD2
    
    

    '''
    # FOR JUST TESTING THE DECODER, WORKS NOW, BUT JUST MUST BE USED MLP INSTEAD OF CNN, 
    # REMEMBER, WE ARE EMULATING DEEPSEQUENCE PAPER IN THIS PART
    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0,trainprocess=False):
        # Encode/decode transformer space
        #pdb.set_trace()

        # Encode/decode semantic space
        #pdb.set_trace()
        mu2, var2 = self.encoder2(x)
        z2 = self.reparameterize(mu2, var2, 1, 1)
        x_mean, x_var = self.decoder2(z2)
        
        return x_mean.unsqueeze(1).contiguous(), \
                x_var.unsqueeze(1).contiguous(), [None, z2], [None, mu2], [None, var2], None
    '''

    '''
    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0,trainprocess=False):
        # Encode/decode transformer space
        #pdb.set_trace()
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), theta_mean, self.Trainprocess, inverse=True)
        self.x_trans1 = x_new
        
        return x_new.contiguous(), \
               None, [z1, None], [mu1, None], [var1, None], x_new, theta_mean
    '''
    #%%
    def sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            x_mean, _ = self.decoder2(z2)
            out_mean = self.stn(x_mean, theta_mean)
            return out_mean

    #%%
    def special_sample(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            x_mean, _ = self.decoder2(z2)
            out_mean = self.stn(x_mean, theta_mean)
            return out_mean, [z1, z2]

    #%%
    def sample_only_trans(self, n, img):
        device = next(self.parameters()).device
        with torch.no_grad():
            img = img.repeat(n, 1, 1, 1).to(device)
            z1 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            out = self.stn(img, theta_mean)
            return out

    #%%
    def sample_only_images(self, n, trans):
        device = next(self.parameters()).device
        with torch.no_grad():
            trans = trans[None, :].repeat(n, 1).to(device)
            z2 = torch.randn(n, self.latent_dim, device=device)
            x_mean, _ = self.decoder2(z2)
            out = self.stn(x_mean, trans)
            return out
    
    #%%
    def sample_transformation(self, n):
        device = next(self.parameters()).device
        with torch.no_grad():
            z1 = torch.randn(n, self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            theta = self.stn.trans_theta(theta_mean.reshape(-1, 2, 3))
            return theta.reshape(-1, 6)
    
    #%%
    def semantics(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1, 1), theta_mean, inverse=True)
        mu2, var2 = self.encoder2(x_new)
        z2 = self.reparameterize(mu2, var2, 1, 1)
        x_mean, x_var = self.decoder2(z2)
        return x_mean, x_var, [z1, z2], [mu1, mu2], [var1, var2]
    
    #%%
    def latent_representation(self, x):
        z_mu1, _ = self.encoder1(x)
        z_mu2, _ = self.encoder2(x)
        return [z_mu1, z_mu2]

    #%%
    def callback(self, writer, loader, epoch):
        n = 10      
        trans = torch.tensor(np.zeros(self.stn.dim()), dtype=torch.float32)
        samples = self.sample_only_images(n*n, trans)
        writer.add_image('samples/fixed_trans', make_grid(samples.cpu(), nrow=n),
                         global_step=epoch)
        del samples
        
        img = (next(iter(loader))[0][0]).to(torch.float32)
        img = img.reshape(-1, *self.input_shape)
        samples = self.sample_only_trans(n*n, img)
        writer.add_image('samples/fixed_img', make_grid(samples.cpu(), nrow=n),
                          global_step=epoch)
        del samples
        
        # Lets log a histogram of the transformation
        theta = self.sample_transformation(1000)
        for i in range(theta.shape[1]):
            writer.add_histogram('transformation/a' + str(i), theta[:,i], 
                                 global_step=epoch, bins='auto')
            writer.add_scalar('transformation/mean_a' + str(i), theta[:,i].mean(),
                              global_step=epoch)
        
        # Also to a decomposition of the matrix and log these values
        if self.stn.dim() == 6:
            values = affine_decompose(theta.view(-1, 2, 3))
            tags = ['sx', 'sy', 'm', 'theta', 'tx', 'ty']
            for i in range(6):
                writer.add_histogram('transformation/' + tags[i], values[i],
                                     global_step=epoch, bins='auto')
                writer.add_scalar('transformation/mean_' + tags[i], values[i].mean(),
                                  global_step=epoch)
            del values
        del theta
            
        # If 2d latent space we can make a fine meshgrid of sampled points
        if self.latent_dim == 2:
            device = next(self.parameters()).device
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            trans = torch.tensor(np.zeros(self.stn.dim()), dtype=torch.float32).repeat(20*20, 1)
            x_mean, x_var = self.decoder2(z.to(device))
            out = self.stn(x_mean, trans.to(device))
            writer.add_image('samples/meshgrid_fixed_trans', make_grid(out.cpu(), nrow=20),
                             global_step=epoch)
            del out
            
            x = np.linspace(-3, 3, 20)
            y = np.linspace(-3, 3, 20)
            z = np.stack([array.flatten() for array in np.meshgrid(x,y)], axis=1)
            z = torch.tensor(z, dtype=torch.float32)
            theta_mean, theta_var = self.decoder1(z.to(device))
            out = self.stn(img.repeat(20*20, 1, 1, 1).to(device), theta_mean)
            writer.add_image('samples/meshgrid_fixed_img', make_grid(out.cpu(), nrow=20),
                             global_step=epoch)
            del out
