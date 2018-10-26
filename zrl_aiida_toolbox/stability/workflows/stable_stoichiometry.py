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

KpointsData = DataFactory('array.kpoints')

from zrl_aiida_toolbox.utils.workflows.change_stoichio import ChangeStoichiometryWorkChain
from zrl_aiida_toolbox.utils.workflows.partial import PartialOccupancyWorkChain
from zrl_aiida_toolbox.stability.workflows.energy_calculation import EnergyWorkchain
# Please improve workchain names
# PartialOccupancyWorkChain = WorkflowFactory('zrl.utils.partial_occ')


# Think about different name: RedoxPotentialsWorkChain
class StableStoichiometryWorkchain(WorkChain):
    
    @classmethod
    def define(cls, spec):    
        super(StableStoichiometryWorkchain, cls).define(spec)
        
        spec.input('input_cif_file_name', valid_type=Str, required=False)
        spec.input('input_structure_aiida', valid_type=StructureData, required=False)
        spec.input('mobile_species', valid_type=Str, required=True)
        spec.input('conf_sampling_method', valid_type=Str, required=True)
        spec.input('sampling_charges', valid_type=ParameterData, required=True)

        spec.outline(
            cls.check_inputs,
            if_(cls.sampling_method_is_MC)(
                cls.generate_structures_MC,
            ).else_(
                cls.no_sampling_method,
            ),
            cls.run_calc,
#            cls.compute_potentials,
            cls.set_output
        )
        
        spec.output_namespace('test', valid_type=Float, dynamic=True)
#        spec.output('phi_red', valid_type=Float)
#        spec.output('phi_ox', valid_type=Float)

        spec.exit_code(1, 'ERROR_MISSING_INPUT_STRUCURE', 'Missing input structure or .cif file.')
        spec.exit_code(2, 'ERROR_AMBIGUOUS_INPUT_STRUCURE', 'More than one input structure or .cif file provided.')
        spec.exit_code(3, 'ERROR_MISSING_SAMPLING_METHOD', 'No valid structure sampling method provided.')

        
    def check_inputs(self):
        if ('input_cif_file_name' not in self.inputs) and ('input_structure_aiida' not in self.inputs):
            return self.exit_codes.ERROR_MISSING_INPUT_STRUCTURE
        if ('input_cif_file_name' in self.inputs) and ('input_structure_aiida' in self.inputs):
            return self.exit_codes.ERROR_AMBIGUOUS_INPUT_STRUCURE
        
    
    def sampling_method_is_MC(self):
        return self.inputs.conf_sampling_method.value == 'sampling_method_MC'

    
    def no_sampling_method(self):
        return self.exit_codes.ERROR_MISSING_SAMPLING_METHOD

    
    def generate_structures_MC(self):
        if ('input_cif_file_name' in self.inputs):
            structure_input_N = StructureData(pymatgen=Structure.from_file(str(self.inputs.input_cif_file_name)))
#        cif = CifParser(self.inputs.cif_file_name)
        if ('input_structure_aiida' in self.inputs):
            structure_input_N = self.inputs.input_structure_aiida

        parameters = ParameterData(dict=dict(charges=sampling_charges))
        
        structures_N = self.submit(PartialOccupancyWorkChain,
                                   structure=structure_input_N,
                                   parameters=parameters)

        structure_input_Np1 = self.submit(ChangeStoichiometryWorkChain,
                                          structure=structure_input_N,
                                          species=self.inputs.mobile_species,
                                          delta_N=1)
        
#         partial_occupancy_structures_Np1 = self.submit(PartialWorkchain,
#                                                    structure=structure_cif,
#                                                    parameters=parameters_partial_occ)

        return ToContext(partial_occupancy_structures_N=partial_occupancy_structures_N)
#                         partial_occupancy_structures_Np1=partial_occupancy_structures_Np1,
#                         partial_occupancy_structures_Nm1=partial_occupancy_structures_Nm1)

    
    def run_calc(self):
        
        settings = ParameterData(dict=dict(gamma_only=True))
        pseudo = Str('sssp')

        options = ParameterData(dict=dict(max_wallclock_seconds=3600,
                              resources=dict(num_machines=1)))

        code = Code.get_from_string('pw.x@daint-tbi')
        
        structures_N = {}
        print len(self.ctx.partial_occupancy_structures_N.get_outputs(link_type=LinkType.RETURN))
        
        print self.ctx.partial_occupancy_structures_N.get_outputs(link_type=LinkType.RETURN)
        
        for i in range(len(self.ctx.partial_occupancy_structures_N.get_outputs(link_type=LinkType.RETURN))):
            key = 'structure_N_%i' % i
            structures_N[key] = self.ctx.partial_occupancy_structures_N.get_outputs(link_type=LinkType.RETURN)[i]

#         structure_1 = StructureData(pymatgen=cif.get_structures()[0])
#         structure_2 = StructureData(pymatgen=cif.get_structures()[0])
#         structures = {
#         structure_1.uuid: structure_1,
#         structure_2.uuid: structure_2
#         }
        
        print structures_N
        energies_N = self.submit(EnergyWorkchain,
                                code=code,
                                structures=structures_N,
                                settings=settings,
                                pseudo_family=pseudo,
                                options=options)
#        return
        return ToContext(energies_N=energies_N)
    
        
    def set_output(self):
        
#        print self.ctx.partial_occupancy_structures_N.get_outputs()[0].get_cell_volume()

        for i in range(len(self.ctx.energies_N.get_outputs())):
            print self.ctx.energies_N.get_outputs()[i]
            
#            energy = self.ctx.energies_N.get_outputs()[i].get_values()

#            self.out('test', Float(self.ctx.phi_red))
#            self.out('test.%i' % i, Float(energy))
            
#        self.out('phi_red', Float(self.ctx.phi_red))
#        self.out('phi_ox', Float(self.ctx.phi_ox))


