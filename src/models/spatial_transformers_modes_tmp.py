import torch

from ..unsuper.unsuper.helper.spatial_transformer import ST_Affine, ST_AffineDecomp, ST_AffineDiff


from ..gp_cpab.src.transformation.gp_cpab import gp_cpab
from ..gp_cpab.src.transformation.libcpab.libcpab import Cpab
#import pdb; pdb.set_trace()

#%%
try:
    from ..gp_cpab.src.transformation.libcpab.libcpab import Cpab

    class ST_CPAB(torch.nn.Module):
        def __init__(self, input_shape):
            super(ST_CPAB, self).__init__()
            self.input_shape = input_shape
            self.cpab = gp_cpab.cpab([2,4], backend='pytorch', device='gpu',
                             zero_boundary=True, 
                             volume_perservation=False)
        
        def forward(self, x, theta, inverse=False):
            if inverse:
                theta = -theta
            out = self.cpab.transform_data(data = x, 
                                           theta = theta,    
                                           outsize = self.input_shape[1:])
            return out
        
        def trans_theta(self, theta):
            return theta
        
        def dim(self):
            return self.cpab.get_theta_dim()
except Exception as e:
    print('Could not import libcpab, error was')
    print(e)
    class ST_CPAB(torch.nn.Module):
        def __init__(self, input_shape):
            super(ST_CPAB, self).__init__()
            self.input_shape = input_shape
            
        def forward(self, x, theta, inverse=False):
            raise ValueError('''libcpab was not correctly initialized, so you 
                             cannot run with --stn_type cpab''')



class ST_GP_CPAB(torch.nn.Module):
  
    def __init__(self, 
                #input_shape, 
                tess_size, config, backend = 'pytorch',
                        device = 'cpu', zero_boundary = False, 
                        volume_perservation = False, override = False, **kargs):

        #import pdb;pdb.set_trace()

        super(ST_GP_CPAB, self).__init__()
        #self.input_shape = input_shape
        gp_params = kargs['argparser_gpdata']
        self.st_gp_cpab = gp_cpab(tess_size, config, backend=backend, device=device, zero_boundary=zero_boundary, 
                                    volume_perservation=volume_perservation, override=override, argparser_gpdata = gp_params)

        
    def setup_ST_GP_CPAB(self, **kargs):
        #import pdb; pdb.set_trace()
        x = kargs['x']
        if 'ref' in kargs:
            ref_msa = kargs['ref']
            outsize = (ref_msa.shape[1], ref_msa.shape[2]) 
        else:
            outsize = (x.shape[1], x.shape[2]) 

    
    def forward(self, x, theta, outsize, inverse=False, **kargs):
        
        if inverse:
            theta = -theta
        
        outmean, out_sampled, forw_per = self.st_gp_cpab.spatial_transformation(x,x,
                                                                        theta,    
                                                                        modeflag = '1D')
        return outmean
        
    def trans_theta(self, theta):
        return theta
    
    def dim(self):
        return self.st_gp_cpab.get_theta_dim()
    
    

def get_transformer(name):
    transformers = {'affine': ST_Affine,
                    'affinediff': ST_AffineDiff,
                    'affinedecomp': ST_AffineDecomp,
                    'cpab': ST_CPAB,
                    'gp_cpab': ST_GP_CPAB
                    }
    assert (name in transformers), 'Transformer not found, choose between: ' \
            + ', '.join([k for k in transformers.keys()])
    return transformers[name]