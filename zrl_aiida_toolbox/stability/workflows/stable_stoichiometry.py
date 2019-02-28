from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory, CalculationFactory
from aiida.work.launch import run
from aiida.work.run import submit

from aiida_quantumespresso.utils.pseudopotential import validate_and_prepare_pseudos_inputs

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure

from datetime import datetime
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
        
        spec.input('structure', valid_type=StructureData, required=False)
        spec.input('verbose', valid_type=Bool, default=Bool(False), required=False)
        
        spec.input('parameters', valid_type=ParameterData)
        
        spec.input('partial_occ_parameters', valid_type=ParameterData, 
                   default=ParameterData(dict=dict(selection='last', n_rounds=200, 
                                                   pick_conf_every=1, n_conf_target=1)))
        
        spec.input_namespace('energy', dynamic=True)
        spec.input('energy.code', valid_type=Code, required=True)
        spec.input('energy.options', valid_type=ParameterData, required=True)
        spec.input_namespace('energy.pseudos', required=False)
        spec.input('energy.pseudo_family', valid_type=Str, required=False)
        spec.input('energy.settings', valid_type=ParameterData, required=False)
        spec.input('energy.kpoints', valid_type=KpointsData)
        spec.input('energy.parameters', valid_type=ParameterData)

        spec.input('seed', valid_type=Int, required=False)

        spec.outline(
            cls.process_inputs,
            cls.generate_supercell,
            # cls.enforce_integer_composition,
            cls.enforce_charge_balance,
            cls.generate_stoichiometries,
            cls.process_structures,
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
        spec.output('seed', valid_type=Int)
        
        spec.output_namespace('structures', dynamic=True)
        spec.output('structures.Nm', valid_type=Str)
        spec.output('structures.N', valid_type=Str)
        spec.output('structures.Np', valid_type=Str)

        spec.exit_code(1, 'ERROR_MISSING_INPUT_STRUCURE', 'Missing input structure or .cif file.')
        spec.exit_code(2, 'ERROR_AMBIGUOUS_INPUT_STRUCURE', 'More than one input structure or .cif file provided.')
        spec.exit_code(3, 'ERROR_MISSING_SAMPLING_METHOD', 'No valid structure sampling method provided.')
        spec.exit_code(4, 'ERROR_CHARGE_BALANCE', 'Incorrect charge balance.')
        spec.exit_code(5, 'ERROR_ERROR_COMPOSITION', 'Incorrect composition.')
        spec.exit_code(6, 'ERROR_ENERGY_WORKCHAIN_WITHOUT_PARAMETER_OUTPUT', 
                       'The provided energy workchain does not return a ParameterData object')
 
    def sampling_method_is_MC(self):
        return self.ctx.conf_sampling_method == 'mc'

    
    def no_sampling_method(self):
        self.report('ERROR: No valid sampling method given.')
        return self.exit_codes.ERROR_MISSING_SAMPLING_METHOD
        
        
    def process_inputs(self):
        self.ctx.seed = self.inputs.seed if 'seed' in self.inputs else Int(np.random.randint(2**31 - 1))
        self.out('seed', self.ctx.seed)
        self.ctx.rs = np.random.RandomState(seed=self.ctx.seed.value)
        
        parameters_dict = self.inputs.parameters.get_dict()
        if ('cif_file' not in parameters_dict) and ('structure' not in self.inputs):
            return self.exit_codes.ERROR_MISSING_INPUT_STRUCTURE
        if ('cif_file' in parameters_dict) and ('structure' in self.inputs):
            return self.exit_codes.ERROR_AMBIGUOUS_INPUT_STRUCURE
        
        if ('cif_file' in parameters_dict):
            self.ctx.structure = Structure.from_file(parameters_dict.get('cif_file'))

        if ('structure' in self.inputs):
            self.ctx.structure = self.inputs.structure.get_pymatgen()
            
        self.ctx.charge_dict = parameters_dict.get('charges')
        
        self.ctx.conf_sampling_method = parameters_dict.get('sampling_method', 'mc')
        self.ctx.stoichiometry_rel_tol = float(parameters_dict.get('stoichiometry_rel_tol', 0.05))
        self.ctx.min_cell_volume = float(parameters_dict.get('min_volume', 1000))
        self.ctx.max_cell_volume = float(parameters_dict.get('max_volume', 5000))
        self.ctx.mobile_species = str(parameters_dict.get('mobile_species', ''))
        self.ctx.num_configurations = int(parameters_dict.get('num_configurations', 1))
        self.ctx.energy_ref = float(parameters_dict.get('energy_ref'))
        
        # TODO : Check all charge are provided against composition.
        # TODO : Check a valid mobile species is provided.
        
        partial_input_dict = self.inputs.partial_occ_parameters.get_dict()
        partial_input_dict['charges'] = self.ctx.charge_dict
        partial_input_dict.setdefault('return_unique', True)
        self.ctx.partial_input = ParameterData(dict=partial_input_dict)
        
        # self.ctx.energy_workchain = 
        # if not any([
        #     "<class 'aiida.orm.data.parameter.ParameterData'>" == p.get('valid_type')
        #     for p in self.ctx.energy_workchain.get_description().get('spec').get('outputs').values()
        # ]):
        #     return self.exit_codes.ERROR_ENERGY_WORKCHAIN_WITHOUT_PARAMETER_OUTPUT
        
        
        # spec.input_namespace('energy.pseudos', required=False)
        # spec.input('energy.pseudo_family', valid_type=Str, required=False)
        
        # for name, input_dict in self.ctx.energy_workchain.get_description().get('spec').get('inputs').items():
        # for name, input_dict in self.ctx.energy_workchain._use_methods.items():
        #     if name in self.inputs.energy and isinstance(self.inputs.energy[name], input_dict.get('valid_types')):
        #         self.ctx.energy_input[name] = self.inputs.energy[name]
        
        self.ctx.structures_N = []
        self.ctx.structures_Np = []
        self.ctx.structures_Nm = []
        
        self.ctx.hashes_N = []
        self.ctx.hashes_Np = []
        self.ctx.hashes_Nm = []
        
        self.ctx.mc_counter = 5
        
    def generate_supercell(self):
        input_composition = self.ctx.structure.composition.as_dict()
        replicate_times = 1.0
        for species in input_composition:
            max_error_current = 0.5/input_composition[species]
            replicate_times = max([replicate_times, max_error_current/float(self.ctx.stoichiometry_rel_tol)])
        
        volume_target = replicate_times * self.ctx.structure.volume
        volume_target = max([self.ctx.min_cell_volume, volume_target])
        volume_target = min([self.ctx.max_cell_volume, volume_target])
        
        return ToContext(replicated=self.submit(ReplicateWorkChain, 
                                                structure=StructureData(pymatgen=self.ctx.structure), 
                                                parameters=ParameterData(dict=dict(min_volume=volume_target))))

    def enforce_charge_balance(self):
        self.ctx.supercell = self.ctx.replicated.get_outputs(StructureData, link_type=LinkType.RETURN)[0].get_pymatgen()
        supercell_composition = self.ctx.supercell.composition.as_dict()
        supercell_total_charge = self.__total_charge(supercell_composition, self.ctx.charge_dict)
        delta_N = - supercell_total_charge / self.ctx.charge_dict[str(self.ctx.mobile_species)]
        
        supercell_balanced = self.submit(ChangeStoichiometryWorkChain, 
                                         structure=StructureData(pymatgen=self.ctx.supercell), 
                                         species=Str(self.ctx.mobile_species), 
                                         delta_N=Float(delta_N), 
                                         distribution=Str('aufbau'))
        
        return ToContext(supercell_balanced=supercell_balanced)

     
    def generate_stoichiometries(self):
        self.ctx.supercell_N = self.ctx.supercell_balanced.get_outputs(StructureData, link_type=LinkType.RETURN)[0].get_pymatgen()
        
        supercell_Np = self.submit(ChangeStoichiometryWorkChain, 
                                   structure=StructureData(pymatgen=self.ctx.supercell_N), 
                                   species=Str(self.ctx.mobile_species), 
                                   delta_N=Float(1.0), 
                                   distribution=Str('aufbau'))
        
        supercell_Nm = self.submit(ChangeStoichiometryWorkChain, 
                                   structure=StructureData(pymatgen=self.ctx.supercell_N), 
                                   species=Str(self.ctx.mobile_species), 
                                   delta_N=Float(-1.0), 
                                   distribution=Str('aufbau'))
        
        return ToContext(supercell_Np=supercell_Np,
                         supercell_Nm=supercell_Nm)
    
    
    def process_structures(self):
        self.ctx.supercell_Np = self.ctx.supercell_Np.get_outputs_dict().get('structure_changed').get_pymatgen()
        self.ctx.supercell_Nm = self.ctx.supercell_Nm.get_outputs_dict().get('structure_changed').get_pymatgen()
       
    def execute_MC(self):
        self.ctx.mc_counter -= 1
        futures = {}
        
        keys = ('Nm', 'N', 'Np')
        n_conf_target = self.ctx.partial_input.get_dict().get('n_conf_target', 1)
        
        for key in keys:
            execute_count = np.ceil((self.ctx.num_configurations - len(self.ctx['structures_%s' % key])) / n_conf_target).astype(int)
            for i in range(execute_count):            
                futures['mc.%s.%d' % (key, i)] = self.submit(PartialOccupancyWorkChain,
                                                             structure=StructureData(pymatgen=self.ctx['supercell_%s' % key]),
                                                             parameters=self.ctx.partial_input,
                                                             seed=Int(self.ctx.rs.randint(2**31 - 1)))
        return ToContext(**futures)
        
    def parse_MC(self):    
        keys = ('Nm', 'N', 'Np')
        new = False
        for i in itertools.count():
            if ('mc.N.%d' % i) not in self.ctx and ('mc.Nm.%d' % i) not in self.ctx and ('mc.Np.%d' % i) not in self.ctx:
                break
            # We parse any result in the context
            for key in keys:
                if ('mc.%s.%d' % (key, i)) in self.ctx:
                    structures = self.ctx['mc.%s.%d' % (key, i)].get_outputs(StructureData, link_type=LinkType.RETURN)
                    del self.ctx['mc.%s.%d' % (key, i)]
                    for structure in structures:
                        if len(self.ctx['structures_%s' % key]) == self.ctx.num_configurations:
                            break
                        if structure.get_hash() not in self.ctx['hashes_%s' % key]:
                            new = True
                            self.ctx['structures_%s' % key].append(structure.get_pymatgen())
                            self.ctx['hashes_%s' % key].append(structure.get_hash())
        
        if new:
            self.ctx.mc_counter += 1
            
        # We continue the loop as long as any category (N, Nm, Np) has fewer than the target number
        # of structure.
        return self.ctx.mc_counter > 0 and any([
            len(self.ctx['structures_%s' % key]) < self.ctx.num_configurations
            for key in keys
        ])
    
    def check_charges_composition(self):
        keys = (('Nm', -1), ('N', 0), ('Np', 1))
        
        composition_ref = self.ctx.supercell_N.composition.element_composition.as_dict()
        
        for key, sign in keys:
            for structure in self.ctx['structures_%s' % key]:
                composition = structure.composition.element_composition.as_dict()
                total_charge = self.__total_charge(composition, self.ctx.charge_dict)
                if np.abs(total_charge - sign * self.ctx.charge_dict[str(self.ctx.mobile_species)]) > 0.001:
                    self.report('ERROR: Incorrect charge balance.')
                    return self.exit_codes.ERROR_CHARGE_BALANCE
                
                continue
                composition_ref_ = copy.deepcopy(composition_ref)
                composition_ref_[self.ctx.mobile_species] += sign
                for species in composition:
                    if np.abs(composition[species] / composition_ref_[species] - 1.0) > self.ctx.stoichiometry_rel_tol: 
                        self.report('ERROR: Incorrect composition.')
                        self.report('Composition %s: %s - %s' % (species, composition.get(species), composition_ref_.get(species)))
                        return self.exit_codes.ERROR_COMPOSITION
            
    def run_calc(self):
        tot_magnetizations = {'Nm': 1, 'N': 0, 'Np': None}
        process = WorkflowFactory('quantumespresso.pw.relax')
        energy_inputs = {
            'code': self.inputs.energy.code,
            'options': self.inputs.energy.options,
            'settings': self.inputs.energy.settings,
            'kpoints': self.inputs.energy.kpoints
        }
        from aiida_quantumespresso.utils.mapping import prepare_process_inputs
        from aiida.orm import Group
        
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        
        futures = {}
        for key, tot_magnetization in tot_magnetizations.items():
            group_name = '%s-%s' % (now, key)
            self.report(group_name)
            group, _ = Group.get_or_create(name=group_name)
            self.out('structures.%s' % key, Str(group_name))
            
            parameters = copy.deepcopy(self.inputs.energy.parameters.get_dict())
            if tot_magnetization is not None:
                parameters.get('SYSTEM')['tot_magnetization'] = tot_magnetization
                
                del parameters.get('SYSTEM')['occupations']
                del parameters.get('SYSTEM')['smearing']
                del parameters.get('SYSTEM')['degauss']
                
            energy_inputs['parameters'] = ParameterData(dict=parameters)
            
            for i, structure in enumerate(self.ctx['structures_%s' % key]):
                self.report(structure)
                structure = StructureData(pymatgen=structure)
                self.out('structures.%s' % structure.uuid, structure)
                group.add_nodes(structure)
                
                inputs = prepare_process_inputs(process, self.inputs.energy)
                futures['energy.%s.%d' % (key, i)] = self.submit(
                    process,
                    structure=structure,
                    pseudo_family=self.inputs.energy.get('pseudo_family', None),
                    max_iterations=Int(1),
                    max_meta_convergence_iterations=Int(1),
                    **energy_inputs
                )
        
        return ToContext(**futures)

    def compute_potentials(self):
        keys = ('Nm', 'N', 'Np')
        energy_min = {
            key: np.inf
            for key in keys
        }
        
        for key in keys:
            for i, structure in enumerate(self.ctx['structures_%s' % key]):
                energy_process = self.ctx['energy.%s.%d' % (key, i)]
                if not energy_process.is_finished_ok:
                    self.report('process %s has failed.' % energy_process.uuid)
                    continue
                energy = [
                    p.get_dict().get('energy') 
                    for p in energy_process.get_outputs(ParameterData, link_type=LinkType.RETURN) 
                    if 'energy' in p.get_dict()
                ][0]
                if energy < energy_min[key]:
                    energy_min[key] = energy
                
        self.out('phi_red', Float(-(energy_min['Np'] - energy_min['N'] - float(self.ctx.energy_ref))))
        self.out('phi_ox',  Float(-(energy_min['N'] - energy_min['Nm'] - float(self.ctx.energy_ref))))
        return
            
    def set_output(self):
        pass
    
    def __total_charge(self, composition, charges):
        total_charge = 0.0
        for species in composition:
            total_charge += composition[species]*charges[species]
        return total_charge
    
    