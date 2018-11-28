from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory
from aiida.work.launch import run
from aiida.work.run import submit

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

import numpy as np

Str = DataFactory('str')
Float = DataFactory('float')
Int = DataFactory('int')

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

KpointsData = DataFactory('array.kpoints')

from zrl_aiida_toolbox.utils.workflows.change_stoichio import ChangeStoichiometryWorkChain
from zrl_aiida_toolbox.utils.workflows.partial import PartialOccupancyWorkChain
from zrl_aiida_toolbox.stability.workflows.energy_calculation import EnergyWorkchain
from zrl_aiida_toolbox.utils.workflows.replicate import ReplicateWorkChain
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
        spec.input('stoichiometry_rel_tol', valid_type=Float, default=Float(0.05), required=False)
        spec.input('min_cell_volume', valid_type=Float, default=Float(1000.0), required=False)
        spec.input('max_cell_volume', valid_type=Float, default=Float(10000.0), required=False)

        spec.outline(
            cls.process_inputs,
            cls.generate_supercell,
            cls.enforce_integer_composition,
            cls.enforce_charge_balance,
            cls.generate_stoichiometries,
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

        
    def process_inputs(self):
        if ('input_cif_file_name' not in self.inputs) and ('input_structure_aiida' not in self.inputs):
            return self.exit_codes.ERROR_MISSING_INPUT_STRUCTURE
        if ('input_cif_file_name' in self.inputs) and ('input_structure_aiida' in self.inputs):
            return self.exit_codes.ERROR_AMBIGUOUS_INPUT_STRUCURE
        
        if ('input_cif_file_name' in self.inputs):
            self.ctx.structure_input = StructureData(pymatgen=Structure.from_file(str(self.inputs.input_cif_file_name)))

        if ('input_structure_aiida' in self.inputs):
            self.ctx.structure_input = self.inputs.input_structure_aiida
        
    def __calculate_factors(self, target_volume, cell_matrix):
        normalized_cell_matrix = 
        volume_norm_cell = 
        target_edge = np.power(target_volume / volume_norm_cell, 1.0/3)
        cell_lengths
        
        n_round = 0
        while do_round:
 
    def calculate_factors(self):
        volume = self.ctx.structure.volume

        if self.ctx.max_electrons:
            electrons = np.floor(sum(n * self.ctx.val_electrons.get(symbol)
                                     for symbol, n in self.ctx.structure.composition.as_dict().items()))
            max_volume = volume * self.ctx.max_electrons / electrons
        else:
            max_volume = self.ctx.max_volume

        max_edge = np.power(max_volume, 1. / 3)
        replicas = np.floor(max_edge / np.diag(self.ctx.structure.lattice.matrix)).astype(int)
        replicas[np.where(replicas == 0)] = 1

        self.ctx.a, self.ctx.b, self.ctx.c = replicas
        
        
    def generate_supercell(self):
        structure_py = self.ctx.structure_input.get_pymatgen()
        input_composition = structure_py.composition.as_dict()
        replicate_times = 1
        for species in input_composition:
            max_error_current = 0.5/input_composition[species]
            replicate_times = max([replicate_times, np.ceil(max_error_current/self.inputs.stoichiometry_rel_tol)])
        
        self.__calculate_factors(replicate_times, self.ctx.structure_input.cell_lengths)
        volume_target = replicate_times * structure_input.get_cell_volume()
        volume_target = max([self.inputs.min_cell_volume, volume_target])
        volume_target = min([self.inputs.max_cell_volume, volume_target])
        parameters = ParameterData(dict=dict(max_volume=volume_target))
        structure_input_scaled = submit(ReplicateWorkChain, structure=self.ctx.structure_input, parameters=parameters)
        
        return ToContext(structure_input_scaled=structure_input_scaled)

#     def generate_supercell(self):
#         replicate_times = 1
#         structure.get_composition()
#         for kind_name in structure.get_composition():
#             kind = structure.get_kind(kind_name)
#   #          for j in range(len(kind.symbols)):
#   #              kind.symbols[j]
#   #              if (kind.symbols[j] == species):
#   #                   weight_species_total += kind.weights[j]
    
    def enforce_integer_composition(self):
        structure_input_scaled=self.ctx.structure_input_scaled.get_outputs_dict()['structure']
        structure_py = structure_input_scaled.get_pymatgen()
        composition = structure_py.composition.as_dict()
        delta
    
    def enforce_charge_balance(self):
        structure=self.ctx.structure_input_Np1.get_outputs_dict()['structure_changed']
        structure_large = result.get('structure')
        
#         structure_py = self.ctx.structure_input.get_pymatgen()
#         input_composition = structure_py.composition.as_dict()
        input_total_charge = self.__total_charge(input_composition, sampling_charges.get_dict()['charges'])
        
        
    def sampling_method_is_MC(self):
        return self.inputs.conf_sampling_method.value == 'sampling_method_MC'

    
    def no_sampling_method(self):
        print('ERROR: No valid sampling method given.')
        return self.exit_codes.ERROR_MISSING_SAMPLING_METHOD


    def generate_stoichiometries(self):        
        structure_input_Np1 = self.submit(ChangeStoichiometryWorkChain, 
                                          structure=self.ctx.structure_input, 
                                          species=self.inputs.mobile_species, 
                                          delta_N=Float(1.0), 
                                          distribution=Str('equal_scale'))
        
        structure_input_Nm1 = self.submit(ChangeStoichiometryWorkChain, 
                                          structure=self.ctx.structure_input, 
                                          species=self.inputs.mobile_species, 
                                          delta_N=Float(-1.0), 
                                          distribution=Str('equal_scale'))
        
        return ToContext(structure_input_Np1=structure_input_Np1,
                         structure_input_Nm1=structure_input_Nm1)
    
    
    def generate_structures_MC(self):
       
        parameters = self.inputs.sampling_charges
        
        structures_N = self.submit(PartialOccupancyWorkChain,
                                   structure=self.ctx.structure_input,
                                   parameters=parameters)

        structures_Np1 = self.submit(PartialOccupancyWorkChain,
                                     structure=self.ctx.structure_input_Np1.get_outputs_dict()['structure_changed'],
                                     parameters=parameters)
        
        structures_Nm1 = self.submit(PartialOccupancyWorkChain,
                                     structure=self.ctx.structure_input_Nm1.get_outputs_dict()['structure_changed'],
                                     parameters=parameters)

        return ToContext(structures_N=structures_N,
                         structures_Np1=structures_Np1,
                         structures_Nm1=structures_Nm1)

    
    def run_calc(self):
        
        settings = ParameterData(dict=dict(gamma_only=True))
        pseudo = Str('sssp')
        options = ParameterData(dict=dict(max_wallclock_seconds=3600,
                                          resources=dict(num_machines=1)))
        code = Code.get_from_string('pw.x@daint-tbi')
        
        structures_N = {}
        print(len(self.ctx.structures_N.get_outputs_dict()))
        print(self.ctx.structures_N.get_outputs_dict())

#         for i in range(len(self.ctx.structures_N.get_outputs_dict())):
#             key = 'structure_N_%i' % i
#             structures_N[key] = self.ctx.partial_occupancy_structures_N.get_outputs(link_type=LinkType.RETURN)[i]

#         print structures_N
#         energies_N = self.submit(EnergyWorkchain,
#                                 code=code,
#                                 structures=structures_N,
#                                 settings=settings,
#                                 pseudo_family=pseudo,
#                                 options=options)

#         return ToContext(energies_N=energies_N)
    
        
    def set_output(self):
        pass
#        print self.ctx.partial_occupancy_structures_N.get_outputs()[0].get_cell_volume()

#         for i in range(len(self.ctx.energies_N.get_outputs())):
#             print self.ctx.energies_N.get_outputs()[i]
            
#            energy = self.ctx.energies_N.get_outputs()[i].get_values()

#            self.out('test', Float(self.ctx.phi_red))
#            self.out('test.%i' % i, Float(energy))
            
#        self.out('phi_red', Float(self.ctx.phi_red))
#        self.out('phi_ox', Float(self.ctx.phi_ox))


    def __count_species(self, structure, species):
        structure_py = structure.get_pymatgen()
        return structure_py.composition.as_dict()[species]


#     def __count_species(self, structure, species):
#         weight_species_total = 0.0
#         for site in structure.sites:
#             kind = structure.get_kind(site.kind_name)
#             for j in range(len(kind.symbols)):
#                 if (kind.symbols[j] == species):
#                     weight_species_total += kind.weights[j]

#         return weight_species_total
    
    
    def __total_charge(self, composition, charges):
        total_charge = 0.0
        for species in composition:
            total_charge += composition[species]*charges[species]
        return total_charge
    
    
    