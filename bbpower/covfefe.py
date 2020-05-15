from bbpipe import PipelineStage
from .types import FitsFile,YamlFile,DummyFile,TextFile

class BBCovFeFe(PipelineStage):
    """
    Template for a covariance matrix stage
    """
    name="BBCovFeFe"
    inputs=[('splits_info',YamlFile),('simulation_info',YamlFile),
            ('mode_coupling_matrix',DummyFile)]
    outputs=[('sims_powspec_list',TextFile),('covariance_matrix',DummyFile)]
    #Should we add a transfer function here?
    config_options={'bpw_edges':[90,110,130],
                    'beam_correct':True,
                    'analytic_covariance':False}

    def run(self) :
        #This stage currently does nothing whatsoever
        print(self.config)
        for inp,_ in self.inputs :
            fname=self.get_input(inp)
            print("Reading "+fname)
            open(fname)

        for out,_ in self.outputs :
            fname=self.get_output(out)
            print("Writing "+fname)
            open(fname,"w")

if __name__ == '__main__':
    cls = PipelineStage.main()
