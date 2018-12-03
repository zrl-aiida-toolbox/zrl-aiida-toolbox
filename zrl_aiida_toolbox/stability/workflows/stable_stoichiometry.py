from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory
from aiida.work.launch import run
from aiida.work.run import submit

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

import numpy as np
import copy

Str = DataFactory('str')
Float = DataFactory('float')
Int = DataFactory('int')
Bool = DataFactory('bool')

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

KpointsData = DataFactory('array.kpoints')

from zrl_aiida_toolbox.utils.workflows.change_stoichio import ChangeStoichiometryWorkChain
from zrl_aiida_toolbox.utils.workflows.partial import PartialOccupancyWorkChain
from zrl_aiida_toolbox.stability.workflows.energy_calculation import EnergyWorkchain
from zrl_aiida_toolbox.utils.workflows.replicate import ReplicateWorkChain


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
        spec.input('energy_ref', valid_type=Float, required=True)
        spec.input('num_configurations', valid_type=Int, required=True)
        spec.input('stoichiometry_rel_tol', valid_type=Float, default=Float(0.05), required=False)
        spec.input('min_cell_volume', valid_type=Float, default=Float(1000.0), required=False)
        spec.input('max_cell_volume', valid_type=Float, default=Float(10000.0), required=False)
        spec.input('verbose', valid_type=Bool, default=Bool(False), required=False)

        spec.outline(
            cls.process_inputs,
            cls.generate_supercell,
#             cls.enforce_integer_composition,
            cls.enforce_charge_balance,
            cls.generate_stoichiometries,
            cls.process_structures,
            if_(cls.sampling_method_is_MC)(
                cls.generate_structures_MC,
            ).else_(
                cls.no_sampling_method,
            ),
            cls.check_charges_composition,
            cls.run_calc,
#            cls.compute_potentials,
            cls.set_output
        )

        spec.output('structure_N', valid_type=StructureData)
        spec.output('structure_Np1', valid_type=StructureData)
        spec.output('structure_Nm1', valid_type=StructureData)
        spec.output_namespace('structures_N', valid_type=StructureData, dynamic=True)
        spec.output_namespace('structures_Np1', valid_type=StructureData, dynamic=True)
        spec.output_namespace('structures_Nm1', valid_type=StructureData, dynamic=True)

#        spec.output_namespace('test', valid_type=Float, dynamic=True)
#        spec.output('phi_red', valid_type=Float)
#        spec.output('phi_ox', valid_type=Float)

        spec.exit_code(1, 'ERROR_MISSING_INPUT_STRUCURE', 'Missing input structure or .cif file.')
        spec.exit_code(2, 'ERROR_AMBIGUOUS_INPUT_STRUCURE', 'More than one input structure or .cif file provided.')
        spec.exit_code(3, 'ERROR_MISSING_SAMPLING_METHOD', 'No valid structure sampling method provided.')
        spec.exit_code(4, 'ERROR_CHARGE_BALANCE', 'Incorrect charge balance.')
        spec.exit_code(5, 'ERROR_ERROR_COMPOSITION', 'Incorrect composition.')

 
    def sampling_method_is_MC(self):
        return self.inputs.conf_sampling_method.value == 'sampling_method_MC'

    
    def no_sampling_method(self):
        print('ERROR: No valid sampling method given.')
        return self.exit_codes.ERROR_MISSING_SAMPLING_METHOD
        
        
    def process_inputs(self):
        if ('input_cif_file_name' not in self.inputs) and ('input_structure_aiida' not in self.inputs):
            return self.exit_codes.ERROR_MISSING_INPUT_STRUCTURE
        if ('input_cif_file_name' in self.inputs) and ('input_structure_aiida' in self.inputs):
            return self.exit_codes.ERROR_AMBIGUOUS_INPUT_STRUCURE
        
        if ('input_cif_file_name' in self.inputs):
            self.ctx.structure_input = StructureData(pymatgen=Structure.from_file(str(self.inputs.input_cif_file_name)))

        if ('input_structure_aiida' in self.inputs):
            self.ctx.structure_input = self.inputs.input_structure_aiida
        self.ctx.charge_dict = self.inputs.sampling_charges.get_dict()['charges']
        
    def generate_supercell(self):
        structure_py = self.ctx.structure_input.get_pymatgen()
        input_composition = structure_py.composition.as_dict()
        replicate_times = 1
        for species in input_composition:
            max_error_current = 0.5/input_composition[species]
            replicate_times = max([replicate_times, np.ceil(max_error_current/float(self.inputs.stoichiometry_rel_tol))])
        
        volume_target = replicate_times * self.ctx.structure_input.get_cell_volume()
        volume_target = max([self.inputs.min_cell_volume, volume_target])
        volume_target = min([self.inputs.max_cell_volume, volume_target])
        
        a, b, c = self.__calculate_factors(volume_target, self.ctx.structure_input.cell)
        self.ctx.structure_input_supercell = StructureData(pymatgen=self.ctx.structure_input.get_pymatgen() \
                                                                       * np.array([a, b, c], dtype=int))
    
    
#     def enforce_integer_composition(self):
#         structure_py = self.ctx.structure_input_supercell.get_pymatgen()
#         composition = self.ctx.structure_input_supercell.get_composition()
#         for species in composition:
#             delta_N = np.round(composition[species]) - composition[species]


    def enforce_charge_balance(self):
        structure_py = self.ctx.structure_input_supercell.get_pymatgen()
        input_supercell_composition = structure_py.composition.as_dict()
        input_supercell_total_charge = self.__total_charge(input_supercell_composition, self.ctx.charge_dict)
        delta_N = -input_supercell_total_charge / self.ctx.charge_dict[str(self.inputs.mobile_species)]
        structure_input_supercell_balanced = self.submit(ChangeStoichiometryWorkChain, 
                                                          structure=self.ctx.structure_input_supercell, 
                                                          species=self.inputs.mobile_species, 
                                                          delta_N=Float(delta_N), 
                                                          distribution=Str('aufbau'))
        return ToContext(structure_input_supercell_balanced=structure_input_supercell_balanced)

     
    def generate_stoichiometries(self):
        self.ctx.structure_input_N = self.ctx.structure_input_supercell_balanced.get_outputs_dict()['structure_changed']
        structure_input_Np1_future = self.submit(ChangeStoichiometryWorkChain, 
                                          structure=self.ctx.structure_input_N, 
                                          species=self.inputs.mobile_species, 
                                          delta_N=Float(1.0), 
                                          distribution=Str('aufbau'))
        
        structure_input_Nm1_future = self.submit(ChangeStoichiometryWorkChain, 
                                          structure=self.ctx.structure_input_N, 
                                          species=self.inputs.mobile_species, 
                                          delta_N=Float(-1.0), 
                                          distribution=Str('aufbau'))
        
        return ToContext(structure_input_Np1_result=structure_input_Np1_future,
                         structure_input_Nm1_result=structure_input_Nm1_future)
    
    
    def process_structures(self):
        self.ctx.structure_input_Np1 = self.ctx.structure_input_Np1_result.get_outputs_dict()['structure_changed']
        self.ctx.structure_input_Nm1 = self.ctx.structure_input_Nm1_result.get_outputs_dict()['structure_changed']
        self.out('structure_N', self.ctx.structure_input_N)
        self.out('structure_Np1', self.ctx.structure_input_Np1)
        self.out('structure_Nm1', self.ctx.structure_input_Nm1)
    
    
    def generate_structures_MC(self):
        parameters = ParameterData(dict=dict(charges=self.ctx.charge_dict,
                          selection='last',
                          n_rounds=10,
                          pick_conf_every=1,
                          n_conf_target=1))
        
        futures = {}
        for i in range(self.inputs.num_configurations.value):
            future = self.submit(PartialOccupancyWorkChain,
                                 structure=self.ctx.structure_input_N,
                                 parameters=parameters)
            futures['N-%d' % i] = future

            future = self.submit(PartialOccupancyWorkChain,
                                         structure=self.ctx.structure_input_Np1,
                                         parameters=parameters)
            futures['Np1-%d' % i] = future

            future = self.submit(PartialOccupancyWorkChain,
                                         structure=self.ctx.structure_input_Nm1,
                                         parameters=parameters)
            futures['Nm1-%d' % i] = future

        return ToContext(**futures)
#                          structures_Np1=structures_Np1_future,
#                          structures_Nm1=structures_Nm1_future)

    
    def check_charges_composition(self):
        structure_py_ref = self.ctx.structure_input_N.get_pymatgen()
        composition_ref_N = structure_py_ref.composition.as_dict()
        
        composition_ref = copy.deepcopy(composition_ref_N)
        self.ctx.structures_N_dict = {}
        i = -1
        for k in range(self.inputs.num_configurations.value):
            self.ctx.structures_N = self.ctx['N-%d' % k]
            self.ctx.structures_Np1 = self.ctx['Np1-%d' % k]
            self.ctx.structures_Nm1 = self.ctx['Nm1-%d' % k]
            
            for j in range(len(self.ctx.structures_N.get_outputs(link_type=LinkType.RETURN))):
                if (type(self.ctx.structures_N.get_outputs(link_type=LinkType.RETURN)[j]) == StructureData):
                    i += 1
                    key = 'structure_N_%i' % i
                    self.ctx.structures_N_dict[key] = self.ctx.structures_N.get_outputs(link_type=LinkType.RETURN)[j]
                    structure_py = self.ctx.structures_N_dict[key].get_pymatgen()
                    composition = structure_py.composition.as_dict()
                    total_charge = self.__total_charge(composition, self.ctx.charge_dict)
                    if np.abs(total_charge) > 0.001:
                        print('ERROR: Incorrect charge balance.')
                        return self.exit_codes.ERROR_CHARGE_BALANCE
                    for species in composition:
                        if np.abs(composition[species]/composition_ref[species] - 1.0) > \
                            float(self.inputs.stoichiometry_rel_tol): 
                            print('ERROR: Incorrect composition.')
                            print('Composition N: ', composition)
                            print('Composition reference N: ', composition_ref)
                            return self.exit_codes.ERROR_COMPOSITION
                        
            composition_ref = copy.deepcopy(composition_ref_N)
            composition_ref[str(self.inputs.mobile_species)] += 1.0
            self.ctx.structures_Np1_dict = {}
            i = -1
            for j in range(len(self.ctx.structures_Np1.get_outputs(link_type=LinkType.RETURN))):
                if (type(self.ctx.structures_Np1.get_outputs(link_type=LinkType.RETURN)[j]) == StructureData):
                    i += 1
                    key = 'structure_Np1_%i' % i
                    self.ctx.structures_Np1_dict[key] = self.ctx.structures_Np1.get_outputs(link_type=LinkType.RETURN)[j]
                    structure_py = self.ctx.structures_Np1_dict[key].get_pymatgen()
                    composition = structure_py.composition.as_dict()
                    total_charge = self.__total_charge(composition, self.ctx.charge_dict)
                    if np.abs(total_charge - self.ctx.charge_dict[str(self.inputs.mobile_species)]) > 0.001:
                        print('ERROR: Incorrect charge balance.')
                        return self.exit_codes.ERROR_CHARGE_BALANCE                
                    for species in composition:
                        if np.abs(composition[species]/composition_ref[species] - 1.0) > \
                            float(self.inputs.stoichiometry_rel_tol):
                            print('ERROR: Incorrect composition.')
                            print('Composition Np1: ', composition)
                            print('Composition reference Np1: ', composition_ref)                        
                            return self.exit_codes.ERROR_COMPOSITION
                        
            composition_ref = copy.deepcopy(composition_ref_N)
            composition_ref[str(self.inputs.mobile_species)] -= 1.0
            self.ctx.structures_Nm1_dict = {}
            i = -1
            for j in range(len(self.ctx.structures_Nm1.get_outputs(link_type=LinkType.RETURN))):
                if (type(self.ctx.structures_Nm1.get_outputs(link_type=LinkType.RETURN)[j]) == StructureData):
                    i += 1
                    key = 'structure_Nm1_%i' % i
                    self.ctx.structures_Nm1_dict[key] = self.ctx.structures_Nm1.get_outputs(link_type=LinkType.RETURN)[j]
                    structure_py = self.ctx.structures_Nm1_dict[key].get_pymatgen()
                    composition = structure_py.composition.as_dict()
                    total_charge = self.__total_charge(composition, self.ctx.charge_dict)
                    if np.abs(total_charge + self.ctx.charge_dict[str(self.inputs.mobile_species)]) > 0.001:
                        print('ERROR: Incorrect charge balance.')
                        return self.exit_codes.ERROR_CHARGE_BALANCE
                    for species in composition:
                        if np.abs(composition[species]/composition_ref[species] - 1.0) > \
                        float(self.inputs.stoichiometry_rel_tol):
                            print('ERROR: Incorrect composition.')
                            print('Composition Nm1: ', composition)
                            print('Composition reference Nm1: ', composition_ref)                        
                            return self.exit_codes.ERROR_COMPOSITION
                    
            
    def run_calc(self):
        settings = ParameterData(dict=dict(gamma_only=True))
        pseudo = Str('SSSP_precision')
        options = ParameterData(dict=dict(max_wallclock_seconds=3600,
                                          resources=dict(num_machines=1),
                                          queue_name='normal',
                                          account='mr28',
                                          custom_scheduler_commands='#SBATCH --constraint=mc'))
        code = Code.get_from_string('QE_pw.x@daint')
        print(self.ctx.structures_N_dict)
        print(self.ctx.structures_Np1_dict)
        print(self.ctx.structures_Nm1_dict)

#         energies_N = self.submit(EnergyWorkchain,
#                                 code=code,
#                                 structures=self.ctx.structures_N_dict,
#                                 settings=settings,
#                                 pseudo_family=pseudo,
#                                 options=options)

#         return ToContext(energies_N=energies_N)


    def compute_potentials(self):
        energies_N = []
        for i in range(len(self.ctx.energies_N.get_outputs(link_type=LinkType.RETURN))):    
            energies_N.append(float(self.ctx.energies_N.get_outputs(link_type=LinkType.RETURN)[i]))        
        energy_N = min(energies_N)
 
        energies_Np1 = []
        for i in range(len(self.ctx.energies_Np1.get_outputs(link_type=LinkType.RETURN))):    
            energies_Np1.append(float(self.ctx.energies_Np1.get_outputs(link_type=LinkType.RETURN)[i]))        
        energy_Np1 = min(energies_Np1)
        
        energies_Nm1 = []
        for i in range(len(self.ctx.energies_Nm1.get_outputs(link_type=LinkType.RETURN))):    
            energies_Nm1.append(float(self.ctx.energies_Nm1.get_outputs(link_type=LinkType.RETURN)[i]))        
        energy_Nm1 = min(energies_Nm1)

        # Make sure all energies in eV
        
        self.ctx.phi_red = -(energy_Np1 - energy_N - float(self.inputs.energy_ref))
        self.ctx.phi_ox = -(energy_N - energy_Nm1 - float(self.inputs.energy_ref))

        
    def set_output(self):
        for key in self.ctx.structures_N_dict:
            self.out('structures_N.%s' % key, self.ctx.structures_N_dict[key])
        for key in self.ctx.structures_Np1_dict:
            self.out('structures_Np1.%s' % key, self.ctx.structures_Np1_dict[key])
        for key in self.ctx.structures_Nm1_dict:
            self.out('structures_Nm1.%s' % key, self.ctx.structures_Nm1_dict[key])
        
#         self.out('phi_red', Float(self.ctx.phi_red))
#         self.out('phi_ox', Float(self.ctx.phi_ox))


    def __calculate_factors(self, target_volume, cell_matrix):
        norm_cell_matrix = []
        cell_lengths = []
        for i in [0,1,2]:
            norm_cell_matrix.append(np.array(cell_matrix[i])/np.linalg.norm(cell_matrix[i]))
            cell_lengths.append(np.linalg.norm(cell_matrix[i]))
        volume_norm_cell = np.linalg.det(norm_cell_matrix)
        target_edge = np.power(target_volume / volume_norm_cell, 1.0/3)
        target_factors = target_edge / np.array(cell_lengths)
        factors = np.floor(target_factors).astype(int)
        
        new_cell_matrix = []
        cell_lengths_new = []
        for i in [0,1,2]:
            new_cell_matrix.append(np.array(cell_matrix[i])*factors[i])
            cell_lengths_new.append(np.linalg.norm(new_cell_matrix[i]))
        volume_new_cell = np.linalg.det(new_cell_matrix)
        n_round = 0
        while volume_new_cell < target_volume:
            n_round += 1
            increase_side = 0
            edge_difference = np.abs(cell_lengths_new[0] + cell_lengths[0] - target_edge)
            for i in [1,2]:
                edge_difference_test = np.abs(cell_lengths_new[i] + cell_lengths[i] - target_edge)
                if edge_difference_test < edge_difference:
                    increase_side = i
                    edge_difference = edge_difference_test
            factors[increase_side] += 1
            for i in [0,1,2]:
                new_cell_matrix[i] = np.array(cell_matrix[i])*factors[i]
                cell_lengths_new[i] = np.linalg.norm(new_cell_matrix[i])
            volume_new_cell = np.linalg.det(new_cell_matrix)
            if n_round >= 10:
                break
 
        return list(factors)


    def __count_species(self, structure, species):
        structure_py = structure.get_pymatgen()
        return structure_py.composition.as_dict()[species]
    
    
    def __total_charge(self, composition, charges):
        total_charge = 0.0
        for species in composition:
            total_charge += composition[species]*charges[species]
        return total_charge
    
    