from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory
from aiida.work.launch import run
from aiida.work.run import submit

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

import numpy as np
import copy, itertools

Str = DataFactory('str')
Float = DataFactory('float')
Int = DataFactory('int')
Bool = DataFactory('bool')

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')

KpointsData = DataFactory('array.kpoints')

from zrl_aiida_toolbox.utils.workflows.change_stoichio import ChangeStoichiometryWorkChain
PartialOccupancyWorkChain = WorkflowFactory('zrl.utils.partial_occ')
ReplicateWorkChain = WorkflowFactory('zrl.utils.replicate')

# Think about different name: RedoxPotentialsWorkChain
class StableStoichiometryWorkChain(WorkChain):
    
    @classmethod
    def define(cls, spec):    
        super(StableStoichiometryWorkChain, cls).define(spec)
        
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
        
        spec.input('partial_occ_parameters', valid_type=ParameterData, 
                   default=ParameterData(dict=dict(selection='last', n_rounds=200, 
                                                   pick_conf_every=1, n_conf_target=1)))
        
        spec.input_namespace('energy', dynamic=True)
        spec.input('energy.code', valid_type=Code, required=True)
        spec.input('energy.workchain', valid_type=Str, required=True)
        spec.input('energy.options', valid_type=ParameterData, required=True)

        spec.outline(
            cls.process_inputs,
            cls.generate_supercell,
            # cls.enforce_integer_composition,
            cls.enforce_charge_balance,
            cls.generate_stoichiometries,
            # cls.process_structures,
            if_(cls.sampling_method_is_MC)(
                while_(cls.parse_MC)(
                    cls.execute_MC
                )
            ).else_(
                cls.no_sampling_method,
            ),
            cls.check_charges_composition,
            cls.run_calc,
            cls.compute_potentials,
            # cls.set_output
        )

        # spec.output('structure_N', valid_type=StructureData)
        # spec.output('structure_Np1', valid_type=StructureData)
        # spec.output('structure_Nm1', valid_type=StructureData)
        # spec.output_namespace('structures_N', valid_type=StructureData, dynamic=True)
        # spec.output_namespace('structures_Np1', valid_type=StructureData, dynamic=True)
        # spec.output_namespace('structures_Nm1', valid_type=StructureData, dynamic=True)

        # spec.output_namespace('test', valid_type=Float, dynamic=True)
        
        spec.output('phi_red', valid_type=Float)
        spec.output('phi_ox', valid_type=Float)

        spec.exit_code(1, 'ERROR_MISSING_INPUT_STRUCURE', 'Missing input structure or .cif file.')
        spec.exit_code(2, 'ERROR_AMBIGUOUS_INPUT_STRUCURE', 'More than one input structure or .cif file provided.')
        spec.exit_code(3, 'ERROR_MISSING_SAMPLING_METHOD', 'No valid structure sampling method provided.')
        spec.exit_code(4, 'ERROR_CHARGE_BALANCE', 'Incorrect charge balance.')
        spec.exit_code(5, 'ERROR_ERROR_COMPOSITION', 'Incorrect composition.')
        spec.exit_code(6, 'ERROR_ENERGY_WORKCHAIN_WITHOUT_PARAMETER_OUTPUT', 
                       'The provided energy workchain does not return a ParameterData object')
 
    def sampling_method_is_MC(self):
        return self.ctx.conf_sampling_method.value == 'sampling_method_MC'

    
    def no_sampling_method(self):
        self.report('ERROR: No valid sampling method given.')
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
        self.ctx.conf_sampling_method = self.inputs.conf_sampling_method
        self.ctx.stoichiometry_rel_tol = float(self.inputs.stoichiometry_rel_tol)
        self.ctx.min_cell_volume = self.inputs.min_cell_volume
        self.ctx.max_cell_volume = self.inputs.max_cell_volume
        self.ctx.mobile_species = self.inputs.mobile_species
        self.ctx.num_configurations = int(self.inputs.num_configurations)
        
        partial_input_dict = self.inputs.partial_occ_parameters.get_dict()
        partial_input_dict['charges'] = self.ctx.charge_dict
        partial_input_dict.setdefault('return_unique', True)
        self.ctx.partial_input = ParameterData(dict=partial_input_dict)
        
        self.ctx.energy_workflow = WorkflowFactory(self.inputs.energy.workchain.value)
        if not any([
            "<class 'aiida.orm.data.parameter.ParameterData'>" == p.get('valid_type')
            for p in self.ctx.energy_workflow.get_description().get('spec').get('outputs').values()
        ]):
            return self.exit_codes.ERROR_ENERGY_WORKCHAIN_WITHOUT_PARAMETER_OUTPUT
        
        self.ctx.energy_input = {}
        for name, input_dict in self.ctx.energy_workflow.get_description().get('spec').get('inputs').items():
            if name in self.inputs.energy and \
               input_dict.get('valid_type') == str(self.inputs.energy[name].__class__):
                self.ctx.energy_input[name] = self.inputs.energy[name]
        
        self.ctx.structures_N = []
        self.ctx.structures_Np = []
        self.ctx.structures_Nm = []
        
        self.ctx.hashes_N = []
        self.ctx.hashes_Np = []
        self.ctx.hashes_Nm = []
        
    def generate_supercell(self):
        structure_py = self.ctx.structure_input.get_pymatgen()
        input_composition = structure_py.composition.as_dict()
        replicate_times = 1.0
        for species in input_composition:
            max_error_current = 0.5/input_composition[species]
            replicate_times = max([replicate_times, max_error_current/float(self.ctx.stoichiometry_rel_tol)])
        
        volume_target = replicate_times * self.ctx.structure_input.get_cell_volume()
        volume_target = max([self.ctx.min_cell_volume.value, volume_target])
        volume_target = min([self.ctx.max_cell_volume.value, volume_target])
        
        return ToContext(replicated=self.submit(ReplicateWorkChain, 
                                                structure=self.ctx.structure_input, 
                                                parameters=ParameterData(dict=dict(min_volume=volume_target))))

    def enforce_charge_balance(self):
        self.ctx.supercell = self.ctx.replicated.get_outputs(StructureData, link_type=LinkType.RETURN)[0]
        supercell_composition = self.ctx.supercell.composition.as_dict()
        supercell_total_charge = self.__total_charge(supercell_composition, self.ctx.charge_dict)
        delta_N = - supercell_total_charge / self.ctx.charge_dict[str(self.inputs.mobile_species)]
        
        supercell_balanced = self.submit(ChangeStoichiometryWorkChain, 
                                         structure=self.ctx.supercell, 
                                         species=self.inputs.mobile_species, 
                                         delta_N=Float(delta_N), 
                                         distribution=Str('aufbau'))
        
        return ToContext(supercell_balanced=supercell_balanced)

     
    def generate_stoichiometries(self):
        self.ctx.supercell_N = self.ctx.supercell_balanced.get_outputs(StructureData, link_type=LinkType.RETURN)[0]
        
        supercell_Np = self.submit(ChangeStoichiometryWorkChain, 
                                   structure=self.ctx.supercell_N, 
                                   species=self.inputs.mobile_species, 
                                   delta_N=Float(1.0), 
                                   distribution=Str('aufbau'))
        
        supercell_Nm = self.submit(ChangeStoichiometryWorkChain, 
                                   structure=self.ctx.supercell_N, 
                                   species=self.inputs.mobile_species, 
                                   delta_N=Float(-1.0), 
                                   distribution=Str('aufbau'))
        
        return ToContext(supercell_Np=supercell_Np,
                         supercell_Nm=supercell_Nm)
    
    
    def process_structures(self):
        self.ctx.supercell_Np = self.ctx.supercell_Np.get_outputs_dict().get('structure_changed')
        self.ctx.supercell_Nm = self.ctx.supercell_Nm.get_outputs_dict().get('structure_changed')
       
    def execute_MC(self):
        futures = {}
        
        keys = ('Nm', 'N', 'Np')
        n_conf_target = self.ctx.partial_input.get_dict().get('n_conf_target', 1)
        
        for key in keys:
            execute_count = np.ceil((self.ctx.num_configurations.value - len(self.ctx['structures_%s' % key])) / n_conf_target)
            for i in range(execute_count):            
                futures['mc.%s.%d' % (key, i)] = self.submit(PartialOccupancyWorkChain,
                                                             structure=self.ctx.['supercell_%s' % key],
                                                             parameters=self.ctx.partial_input)
        return ToContext(**futures)
    
    def parse_MC(self):    
        keys = ('Nm', 'N', 'Np')
        for i in itertools.count():
            if ('mc.N.%d' % i) not in self.ctx and ('mc.Nm.%d' % i) not in self.ctx and ('mc.Np.%d' % i) not in self.ctx:
                break
            # We parse any result in the context
            for key in keys:
                if ('mc.%s.%d' % (key, i)) in self.ctx:
                    structures = self.ctx['mc.%s.%d' % (key, i)].get_outputs(StructureData, link_type=LinkType.RETURN)
                    del self.ctx['mc.%s.%d' % (key, i)]
                    for structure in structures:
                        if len(self.ctx['structures_%s' % key]) == self.ctx.num_configurations.value:
                            break
                        if structure.get_hash() not in self.ctx['hashes_%s' % key]:
                            self.ctx['structures_%s' % key].append(structure)
                            self.ctx['hashes_%s' % key].append(structure.get_hash())
        # We continue the loop as long as any category (N, Nm, Np) has fewer than the target number
        # of structure.
        return any([
            len(self.ctx['structures_%s' % key]) < self.ctx.num_configurations.value
            for key in keys
        ])
    
    def check_charges_composition(self):
        keys = (('Nm', -1), ('N', 0), ('Np', 1))
        
        structure_py_ref = self.ctx.supercell_N.get_pymatgen()
        composition_ref_N = structure_py_ref.composition.as_dict()
        
        for key, sign in keys:
            for structure in self.ctx['structures_%s' % key]:
                composition = structure.get_pymatgen().composition.as_dict()
                total_charge = self.__total_charge(composition, self.ctx.charge_dict)
                if np.abs(total_charge - sign * self.ctx.charge_dict[str(self.inputs.mobile_species)]) > 0.001:
                    self.report('ERROR: Incorrect charge balance.')
                    return self.exit_codes.ERROR_CHARGE_BALANCE
                
                for species in composition:
                    if np.abs(composition[species]/composition_ref[species] - 1.0) > self.ctx.stoichiometry_rel_tol: 
                        self.report('ERROR: Incorrect composition.')
                        self.report('Composition %s: %s' % (key, composition))
                        self.report('Composition reference %s: %s', (key, composition_ref))
                        return self.exit_codes.ERROR_COMPOSITION
            
    def run_calc(self):
        keys = ('Nm', 'N', 'Np')
        futures = {}
        for key in keys:
            for i, structure in enumerate(self.ctx['structures_%s' % key]):
                futures['energy.%s.%d' % (key, i)] = self.submit(
                    self.ctx.energy_workchain,
                    structure=structure,
                    **self.ctx.energy_input
                )
                
        return ToContext(**futures)


    def compute_potentials(self):
        keys = ('Nm', 'N', 'Np')
        energy_min = {
            key: np.inf
            for key in keys
        }
        
        for key, sign in keys:
            for i, structure in enumerate(self.ctx['structures_%s' % key]):
                energy = [
                    p.get_dict().get('energy') 
                    for p in self.ctx[key_m].get_outputs(ParameterData, link_type=LinkType.RETURN) 
                    if 'energy' in p.get_dict()
                ][0]
                if energy < energy_min[key]:
                    energy_min[key] = energy
                
        self.out('phi_red', Float(-(energy_min['Np'] - energy_min['N'] - float(self.inputs.energy_ref))))
        self.out('phi_ox',  Float(-(energy_min['N'] - energy_min['Nm'] - float(self.inputs.energy_ref))))
        return
            
    def set_output(self):
        pass
    
    def __total_charge(self, composition, charges):
        total_charge = 0.0
        for species in composition:
            total_charge += composition[species]*charges[species]
        return total_charge
    
    