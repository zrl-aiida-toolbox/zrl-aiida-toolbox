from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory
from aiida.work.launch import run
from aiida.work.run import submit

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

Str = DataFactory('str')
Float = DataFactory('float')
Int = DataFactory('int')

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')


class ChangeStoichiometryWorkChain(WorkChain):
    
    @classmethod
    def define(cls, spec):    
        super(ChangeStoichiometryWorkChain, cls).define(spec)
        
        spec.input('structure', valid_type=StructureData, required=True)
        spec.input('species', valid_type=Str, required=True)
        spec.input('delta_N', valid_type=Int, required=True)

        spec.outline(
        )
        
        spec.output_namespace('test', valid_type=Float, dynamic=True)

