from bbpipe import PipelineStage
from .types import FitsFile, TextFile
import numpy as np

class BBMaskPreproc(PipelineStage):
    """
    Template for a mask pre-processing stage
    """
    name='BBMaskPreproc'
    inputs= [('binary_mask',FitsFile),('source_data',TextFile)]
    outputs=[('window_function',FitsFile)]
    config_options={'aposize_edges':1.0,
                    'apotype_edges':'C1',
                    'aposize_srcs':0.1,
                    'apotype_srcs':'C1'}

    def run(self) :
        #Read input mask
        import healpy as hp #We will want to be more general than assuming HEALPix
        mask_raw=hp.read_map(self.get_input('binary_mask'),verbose=False)

        #Read point source data
        #Right now this is a simple text file, but this is probably not ideal.
        ps_ra,ps_dec,ps_size=np.loadtxt(self.get_input('source_data'),unpack=True,ndmin=2)

        #Now we should do stuff (apodization, inverse-variance weighting etc.)
        #but this is just a placeholder
        print("BBMaskPreproc currently does nothing")

        #Write window function
        #Currently just writing the input mask.
        hp.write_map(self.get_output('window_function'),mask_raw,overwrite=True)

if __name__ == '__main__':
    cls = PipelineStage.main()
