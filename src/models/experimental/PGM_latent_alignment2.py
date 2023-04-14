from ..vitae_ci_gp_tmp import *
from ..encoder_decoder import conv_attention, conv_attention2
import numpy as np
PI = torch.from_numpy(np.asarray(np.pi))


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1)   

class PGM_latent_alignment(VITAE_CI):

    def __init__(self, input_shape, config, latent_dim, encoder, decoder, outputdensity, ST_type, **kwargs):
        super(VITAE_CI, self).__init__()
        # Constants

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = latent_dim
        self.outputdensity = outputdensity
        self.alphabet = kwargs["alphabet_size"]
        ndim, device, gp_params = kwargs["trans_parameters"]

        # Spatial transformer
        
        self.ST_type = ST_type

        self.stn = get_transformer(ST_type)(ndim, config, backend='pytorch', device=device, zero_boundary=False,
                                          volume_perservation=False, override=False, argparser_gpdata = gp_params)

        self.Trainprocess = True

        self.diag_domain = kwargs['diagonal_att_regions']
        #self.attention =  conv_attention(channel_shape=abs(self.diag_domain[0]) + abs(self.diag_domain[1] + 1), 
        #                                shape_signal=self.input_shape, kernel=3)

        self.attention =  conv_attention2(channel_shape= np.sum(np.absolute(self.diag_domain))+1, 
                                         input_shape = self.input_shape,
                                        shape_signal=self.input_shape[0],
                                        kernel = 5,#kernel= 21, #12 Always in uneven number - impar, otherwise will increase the final length -- 11 to 15
                                        deconv_flat_out=self.stn.dim())
        
        '''
        # Define encoder and decoder
        if isinstance(encoder,tuple) and isinstance(decoder,tuple):
            self.encoder1 = encoder[0](input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder1 = decoder[0]((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet)
            
        else:
            self.encoder1 = encoder(input_shape, latent_dim, layer_ini = self.alphabet)
            self.decoder1 = decoder((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet)
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
            
    def get_deepsequence_nograd(self, x, DS):
        #x_copy = torch.tensor(x, requires_grad=False)
        with torch.no_grad():
            DS.eval()
            x_mean_no_grad, x_var_no_grad,_,__,____,_____ = DS(x)
        return x_mean_no_grad, x_var_no_grad

    def MC_sampling_DeepSequence(self, DS, iters=100):
        DS.eval()
        set_of_samples = DS.sample(iters)[0] #[ DS.sample(1)[0] for i in range(0,iters)]
        MC_sample = torch.mean(set_of_samples, dim=0)
        return MC_sample

    '''
    def get_diagonal_attention(self, Matrix, comp, min_r, max_r, list_attention = []):
        # Works assuming that we are dealing with square matrixes
        if comp <= max_r and comp >= min_r:
            list_attention.append( torch.cat( (torch.tensor([0.0]*abs(comp)) , torch.diagonal(Matrix, comp)) ) )
            self.get_diagonal_attention(Matrix, comp-1, min_r, max_r, list_attention)

    def get_batch_diagonal_attention(self, Matrix, min_r, max_r):
        list_batch = []
        list_attention = []
        comp = max_r
        for m in Matrix:
            self.get_diagonal_attention(m, comp, min_r, max_r, list_attention )
            list_batch.append( torch.stack(list_attention) )
            list_attention.clear()

        batch_diag_attention = torch.stack(list_batch)
        return batch_diag_attention
    '''
    def get_diagonal_attention(self, Matrix, comp, min_r, max_r, list_attention = []):
    # Works assuming that we are dealing with square matrixes

        if comp <= max_r and comp >= min_r:
            prod_diag = torch.diagonal(Matrix, comp)
            if comp > 0 and len(prod_diag) < len(Matrix):
                list_attention.append( torch.cat( 
                    ( prod_diag, torch.tensor([0]* ( len(Matrix) - len(prod_diag)) ) )
                    ) )
            elif comp < 0 and len(prod_diag) < len(Matrix):
                list_attention.append( torch.cat( 
                    ( torch.tensor([0]* (len(Matrix) - len(prod_diag) )), prod_diag ) # change 0 for prod_diag.min()
                    ) )
            else:
                list_attention.append( torch.diagonal(Matrix, comp) )
            
            self.get_diagonal_attention(Matrix, comp-1, min_r, max_r, list_attention)


    def get_batch_diagonal_attention(self, Matrix, comp, min_r, max_r):
        list_batch = []
        list_attention = []

        for m in Matrix:
            if max_r == None or min_r == None:
                list_batch.append( m.T )
            else:
                self.get_diagonal_attention(m, comp, min_r, max_r, list_attention )
                list_batch.append(   torch.fliplr(torch.column_stack(list_attention) ).T  )
            list_attention.clear()

        batch_diag_attention = torch.stack(list_batch)
        return batch_diag_attention

    def get_attention_matrix(self, raw_seqs, DS, iters=100):

        MC_DS_protein = self.MC_sampling_DeepSequence( DS, iters)
        list_of_attentions = []

        for cont, prot in enumerate(raw_seqs):
            seq = torch.matmul(prot, MC_DS_protein.T)
            list_of_attentions.append(seq)
        return list_of_attentions



    def forward(self, x, deepS, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space

        # extract the attention mechanisms between the
        l_attention = self.get_attention_matrix(x,deepS, iters=1000)#1000
        batch_diagonal_regions =  self.get_batch_diagonal_attention(l_attention, self.diag_domain[1], self.diag_domain[0], self.diag_domain[1])
        #batch_diagonal_regions =  self.get_batch_diagonal_attention(l_attention, self.diag_domain[1], None, None)

        # number_channels = sup_diagonal+inf_diagonal+1, #samples = amount of aminos
        # Goal: try to get the best path to achieve the features to guide the alignment through theta.
        #       That's why it is important to transpose or permute the signal, in order to make the
        #       channels as the amount of diagonal components while the amount of samples should be the 
        #       aminoacids 
        '''batch_diagonal_regions = torch.log(batch_diagonal_regions)
        batch_diagonal_regions = batch_diagonal_regions.nan_to_num(nan=-100)'''
        

        theta_mean = self.attention(batch_diagonal_regions)
        #theta_mean = self.attention(batch_diagonal_regions)


        '''
        mu1, var1 = self.encoder1(attention_repr)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)
        KLD = self.KL(z1, mu1, var1)
        '''

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), theta_mean, self.Trainprocess, inverse=True)

        # In case of using the log space in the prior for avoid local minima 
        #if self.prior_space=='log':
        #    x_new = self.outputnonlin(x_new)
        '''-------------------------------------------------------------------------------------------------------'''
        # Pretrained DeepSequence Output
        x_mean_no_grad, x_var_no_grad = self.get_deepsequence_nograd(x_new,deepS)

        x_mean = torch.tensor(x_mean_no_grad, requires_grad=True)
        x_var = torch.tensor(x_var_no_grad, requires_grad=True)
        '''-------------------------------------------------------------------------------------------------------'''
        # "Detransform" output
        self.stn.st_gp_cpab.interpolation_type = 'GP' 
        x_mean = self.stn(x_mean, theta_mean,  self.Trainprocess, inverse=False)
        x_var = self.stn(x_var, theta_mean, self.Trainprocess, inverse=False)
        x_var = switch*x_var + (1-switch)*0.02**2

        # In case of using the log space in the prior for avoid local minima 
        #if self.prior_space=='log':
        #    x_mean = self.outputnonlin(x_mean)
        #    x_var = self.outputnonlin(x_var)
        
        return x_mean.contiguous(), \
                x_var.contiguous(), [None, None], [None, None], [None, None], x_new, theta_mean, None


    def sample_only_trans(self, x):
        device = next(self.parameters()).device
        with torch.no_grad():
            mu1, var1 = self.encoder1(x)
            z1 = torch.randn(x.shape[0], self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            self.stn.st_gp_cpab.interpolation_type = 'GP'
            x_new = self.stn(x.repeat(1, 1, 1), theta_mean, self.Trainprocess, inverse=True)
            return x_new, theta_mean