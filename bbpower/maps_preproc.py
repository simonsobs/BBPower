from bbpipe import PipelineStage
from .types import FitsFile, YamlFile, DummyFile
import yaml
import numpy as np

class BBMapsPreproc(PipelineStage):
    """
    Template for a map pre-processing stage
    """
    name="BBMapsPreproc"
    inputs=[('splits_info',YamlFile),('window_function',FitsFile)]
    outputs=[('nmt_fields',DummyFile)]
    config_options={'purify_b':False}

    def run(self) :
        import healpy as hp #We will be more general than just assuming HEALPix
        cfg_maps=yaml.load(open(self.get_input('splits_info')))

        size_info=cfg_maps['DataSize']
        npix=hp.nside2npix(size_info['nside'])
        n_splits=size_info['nsplits']
        n_nu=size_info['nfreq']

        #Read window function
        window=hp.read_map(self.get_input('window_function'),verbose=False)
        if len(window)!=npix :
            raise ValueError("Window function has wrong pixelization")

        #Read maps
        #Map dimensions: [n_nu, n_splits, n_pol, n_pix]
        maps_all=np.zeros([n_nu,n_splits,2,npix])

        for inu in range(n_nu) :
            for isplit in range(n_splits) :
                fname=cfg_maps['Maps']['prefix']
                fname+="_split%dof%d_nu%dof%d.fits"%(isplit+1,n_splits,
                                                     inu+1,n_nu)
                print("Reading split: "+fname)
                maps_all[inu,isplit,:,:]=np.array(hp.read_map(fname,verbose=False,
                                                              field=[1,2]))

        #Now we want to do stuff with the maps (purify, filter, etc.)
        #For now we don't do anything at all
        print("BBMapsPreproc currently does nothing")
        
        #Write output
        #Right now we just write the raw input, but we probably want to end up some
        #set of purified a_lms + metadata
        hp.write_map(self.get_output('nmt_fields'),
                     maps_all.reshape(n_nu*n_splits*2,npix),
                     overwrite=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
