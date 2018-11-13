from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory
from aiida.work.launch import run
from aiida.work.run import submit


StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
Str = DataFactory('str')
KpointsData = DataFactory('array.kpoints')
Float = DataFactory('float')

class EnergyWorkchain(WorkChain):
    
    @classmethod
    def define(cls, spec):    
        super(EnergyWorkchain, cls).define(spec)
        
        spec.input('code', valid_type=Code)
        spec.input_namespace('structures', valid_type=StructureData, dynamic=True)
        spec.input('parameters', valid_type=ParameterData, required=False)
        spec.input('settings', valid_type=ParameterData, required=False)
        spec.input('kpoints', valid_type=KpointsData, required=False)
        spec.input('pseudo_family', valid_type=Str, required=False)
        spec.input('options', valid_type=ParameterData, required=False)
        
        spec.outline(
            cls.check_inputs,
            cls.run_calc,
            cls.set_output
        )
        
        spec.output_namespace('energy', valid_type=Float, dynamic=True)

    def check_inputs(self):
        self.ctx.parameters = self.inputs.parameters \
            if 'parameters' in self.inputs \
            else ParameterData(dict=dict(CONTROL=dict(calculation='scf',
                                                      restart_mode='from_scratch',
                                                      wf_collect=True),
                                         SYSTEM=dict(ecutwfc=30.,
                                                     ecutrho=240.),
                                         ELECTRONS=dict(conv_thr=1.e-6)))
        
        if 'kpoints' in self.inputs:
            self.ctx.kpoints = self.inputs.kpoints
        else: 
            self.ctx.kpoints = KpointsData()
            self.ctx.kpoints.set_kpoints_mesh([1, 1, 1])
        
    def run_calc(self):
        futures = {}
        for key, structure in self.inputs.structures.items():
            futures[key] = self.submit(PwBaseWorkChain,
                                       code=self.inputs.code,
                                       structure=structure,
                                       parameters=self.ctx.parameters,
                                       settings=self.inputs.settings,
                                       kpoints=self.ctx.kpoints,
                                       pseudo_family=self.inputs.pseudo_family,
                                       options=self.inputs.options)
        return ToContext(**futures)
    
    def set_output(self):
        for key in self.inputs.structures.keys():
            result = self.ctx[key].get_outputs(node_type=ParameterData)[0].get_dict()
            self.out('energy.%s' % key, Float(result.get('energy')))
        
        
