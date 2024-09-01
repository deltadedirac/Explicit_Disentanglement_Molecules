from ..vitae_ci_gp_tmp import *
PI = torch.from_numpy(np.asarray(np.pi))
import torch.distributions as D


class vitae_ci_gp_no_deepseq(VITAE_CI):

    def __init__(self, input_shape, config, latent_dim, encoder, decoder, outputdensity, ST_type, **kwargs):
        super(VITAE_CI, self).__init__()
        # Constants

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = latent_dim
        self.outputdensity = outputdensity
        self.alphabet = kwargs["alphabet_size"]
        ndim, device, gp_params = kwargs["trans_parameters"]

        self.prior = D.Normal(torch.zeros(latent_dim).to(device), 
                              torch.ones(latent_dim).to(device))

        # Spatial transformer
        
        self.ST_type = ST_type

        self.stn = get_transformer(ST_type)(ndim, config, backend='pytorch', device=device, zero_boundary=False,
                                          volume_perservation=False, override=False, argparser_gpdata = gp_params)
        
        if 'posterior_variance' in kwargs:
            self.stn.st_gp_cpab.set_posterior_variance(kwargs['posterior_variance'])

        self.Trainprocess = True
        self.x_trans1 = torch.tensor([])

        #self.outputnonlin = nn.Softmax(dim=2)

        # Define encoder and decoder
        if isinstance(encoder,tuple) and isinstance(decoder,tuple):
            self.encoder1 = encoder[0](input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder1 = decoder[0]((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet)
            
        else:
            self.encoder1 = encoder(input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder1 = decoder((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet)

    def log_normal_diag(self, x, mu, log_var, reduction=None, dim=None):
        D = x.shape[1]
        log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p
    
    def set_activator_for_STlayer(self, space='log'):
        self.prior_space=space
    
    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:].sqrt() * eps).reshape(-1, latent_dim)
        
        #return D.Normal(mu,  var.sqrt()).rsample()


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
        return torch.nn.functional.kl_div( log_qz, log_z, reduction='none', log_target =True)
        
        # Due to the scarse
        """
        q_dist = D.Normal(mu, log_var.sqrt())
        kl = D.kl_divergence(q_dist, self.prior)
        return kl.mean(-1)
        """
        """
        prior = D.Normal(torch.zeros_like(q_dist.mean), 
                         torch.ones_like(q_dist.variance))
        kl = D.kl_divergence(q_dist, prior).mean(-1)
        """
        
            
    @torch.no_grad()
    def get_deepsequence_nograd(self, x, DS):
        #x_copy = torch.tensor(x, requires_grad=False)
        with torch.no_grad():
            DS.eval()
            x_mean_no_grad, x_var_no_grad,_,__,____,KLds = DS(x) #_copy)
        return x_mean_no_grad, x_var_no_grad, KLds

    def forward(self, x, deepS, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space
        #import ipdb; ipdb.set_trace()
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)
        KLD = self.KL(z1, mu1, var1)

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), theta_mean, self.Trainprocess, inverse=True)


        '''-------------------------------------------------------------------------------------------------------'''
        # Pretrained DeepSequence Output
        x_mean, x_var, KLds = self.get_deepsequence_nograd(x_new,deepS)

        '''-------------------------------------------------------------------------------------------------------'''
        # "Detransform" output
        self.stn.st_gp_cpab.interpolation_type = 'GP' 
        x_mean = self.stn(x_mean, theta_mean,  self.Trainprocess, inverse=False)
        x_var = self.stn(x_var, theta_mean, self.Trainprocess, inverse=False)

        
        return x_mean.contiguous(), \
                x_var.contiguous(), [z1, None], [mu1, None], [var1, None], x_new, theta_mean, KLD, KLds


    def sample_only_trans(self, x):
        device = next(self.parameters()).device
        with torch.no_grad():
            mu1, var1 = self.encoder1(x)
            z1 = torch.randn(x.shape[0], self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            self.stn.st_gp_cpab.interpolation_type = 'GP'
            x_new = self.stn(x.repeat(1, 1, 1), theta_mean, self.Trainprocess, inverse=True)
            return x_new, theta_mean