from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory, CalculationFactory
from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida.work.launch import run
from aiida.work.run import submit
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_utils.pseudos import get_pseudos

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, SpeciesComparator, ElementComparator

from copy import deepcopy
import numpy as np
import itertools
import re
import glob
import os


Str = DataFactory('str')
Float = DataFactory('float')
Int = DataFactory('int')
Bool = DataFactory('bool')

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ListData = DataFactory('list')

KpointsData = DataFactory('array.kpoints')

PartialOccupancyWorkChain = WorkflowFactory('zrl.utils.partial_occ')
ReplicateWorkChain = WorkflowFactory('zrl.utils.replicate')


class PhaseStabilityWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):    
        super(PhaseStabilityWorkChain, cls).define(spec)
        
        spec.input('input_cif_folder', valid_type=Str)
        spec.input('element_list', valid_type=ListData)
        spec.input('oxidation_states', valid_type=ParameterData)
        
        spec.input('stoichiometry_rel_tol', valid_type=Float, default=Float(0.05))
        spec.input('min_cell_volume', valid_type=Float, default=Float(1000.0))
        spec.input('max_cell_volume', valid_type=Float, default=Float(1500.0))
        
        spec.input('seed', valid_type=Int, required=False)
        spec.input('mc_temp', valid_type=Float, default=Float(1000.0))
        spec.input('equilibration', valid_type=Int, default=Int(5000))
        
        spec.input_namespace('energy', dynamic=True)
        spec.input('energy.code', valid_type=Code, required=True)
        spec.input('energy.options', valid_type=ParameterData, required=True)
        spec.input_namespace('energy.pseudos', required=False)
        spec.input('energy.pseudo_family', valid_type=Str, required=False)
        spec.input('energy.settings', valid_type=ParameterData, required=False)
        spec.input('energy.kpoints', valid_type=KpointsData)
        spec.input('energy.parameters', valid_type=ParameterData)
        
        spec.input('mobile_species', valid_type=Str)
        spec.input('main_composition', valid_type=ParameterData)
        spec.input('ref_composition', valid_type=ParameterData)
        spec.input('max_num_reactions', valid_type=Int, default=Int(100000000))
        spec.input('det_thr', valid_type=Float, default=Float(1e-6))
        spec.input('coeff_thr', valid_type=Float, default=Float(1e-6))
        spec.input('energy_supercell_error', valid_type=Float, default=Float(1.0))
        spec.input('error_tol_potential', valid_type=Float, required=False)

        spec.outline(
            cls.initialize,
            cls.remove_multiple_cif,
            cls.generate_supercells,
            cls.process_supercells,
            cls.generate_configurations,
            cls.process_configurations,
            cls.run_calc,
            cls.process_calc,
            cls.compute_potentials
        )

        spec.output('seed', valid_type=Int)
        
        spec.output('structures_match_not_used', valid_type=ParameterData)
        
        spec.output_namespace('structure', dynamic=True)
        spec.output_namespace('properties', dynamic=True)
        
        spec.output('structures_min_energy', valid_type=ParameterData)
        
        spec.output('potentials_all', valid_type=ParameterData)
        spec.output('decomp_relevant', valid_type=ParameterData)
        
        spec.exit_code(1, 'ERROR_MAX_NUM_REACTIONS', 'Too many decomposition reactions to compute.')
        
        
    def initialize(self):
        self.ctx.input_cif_folder = self.inputs.input_cif_folder.value
        self.ctx.element_list = self.inputs.element_list.get_list()
        self.ctx.oxidation_states = self.inputs.oxidation_states.get_dict()
        
        self.ctx.structures_used = {}
        self.ctx.structures_match_not_used = {}
        self.ctx.structures_min_energy = {}
        
        self.ctx.stoichiometry_rel_tol = self.inputs.stoichiometry_rel_tol.value
        self.ctx.min_cell_volume = self.inputs.min_cell_volume.value
        self.ctx.max_cell_volume = self.inputs.max_cell_volume.value
        
        self.ctx.seed = self.inputs.seed if 'seed' in self.inputs else Int(np.random.randint(2**31 - 1))
        self.out('seed', self.ctx.seed)
        self.ctx.rs = np.random.RandomState(seed=self.ctx.seed.value)
        
        self.ctx.mc_temp = self.inputs.mc_temp.value
        self.ctx.equilibration = self.inputs.equilibration.value
        
        self.ctx.pseudo_family = self.inputs.energy.get('pseudo_family', None)
        self.ctx.energy_inputs = {
            'code': self.inputs.energy.code,
            'options': self.inputs.energy.options,
            'settings': self.inputs.energy.settings,
            'kpoints': self.inputs.energy.kpoints
        }
        self.ctx.energy_parameters = self.inputs.energy.parameters.get_dict()
        
        self.ctx.mobile_species = self.inputs.mobile_species.value
        self.ctx.main_composition = self.inputs.main_composition.get_dict()
        self.ctx.ref_composition = self.inputs.ref_composition.get_dict()        
        self.ctx.max_num_reactions = self.inputs.max_num_reactions.value
        self.ctx.det_thr = self.inputs.det_thr.value
        self.ctx.coeff_thr = self.inputs.coeff_thr.value
        self.ctx.energy_supercell_error = self.inputs.energy_supercell_error.value
        self.ctx.error_tol_potential = self.inputs.error_tol_potential.value if 'error_tol_potential' in self.inputs else None
        
        
    def remove_multiple_cif(self):
        structMatch = StructureMatcher(ltol=0.05,
                                   stol=0.05,
                                   angle_tol=2,
                                   primitive_cell=False,
                                   scale=False,
                                   attempt_supercell=False,
                                   allow_subset=False,
                                   comparator=ElementComparator(),
                                   supercell_size='num_sites',
                                   ignored_species=None)
        
        for input_cif_file_path in glob.glob(self.ctx.input_cif_folder + '/*cif'):
            input_cif_file = os.path.basename(input_cif_file_path).replace('.', '_')
            use_cif = True
            try:
                structure = Structure.from_file(input_cif_file_path)
            except:
                print('Structure import from cif not successful.')
                continue
            
            structure.remove_oxidation_states()

            for element in structure.composition.element_composition.elements:
                if str(element) not in self.ctx.element_list:
                    use_cif = False

            for key in self.ctx.structures_used:
                if structMatch.fit(structure, self.ctx.structures_used[key]['structure_original']):
                    use_cif = False
                    self.ctx.structures_match_not_used[input_cif_file] = key

            if use_cif:
                self.ctx.structures_used[input_cif_file] = {}
                self.ctx.structures_used[input_cif_file]['structure_original'] = structure

        self.out('structures_match_not_used', ParameterData(dict=self.ctx.structures_match_not_used))

        
    def generate_supercells(self):
        futures = {}
        for input_cif_file in self.ctx.structures_used:
            structure = self.ctx.structures_used[input_cif_file]['structure_original']
            
            input_composition = structure.composition.as_dict()
            replicate_times = 1.0
            for species in input_composition:
                max_error_current = 0.5/input_composition[species]
                replicate_times = max([replicate_times, max_error_current/self.ctx.stoichiometry_rel_tol])

            volume_target = replicate_times * structure.volume
            volume_target = max([self.ctx.min_cell_volume, volume_target])
            volume_target = min([self.ctx.max_cell_volume, volume_target])

            futures['replicate.%s' % (input_cif_file)] = self.submit(ReplicateWorkChain, 
                                                    structure=StructureData(pymatgen=structure), 
                                                    parameters=ParameterData(dict=dict(min_volume=volume_target)))
        
        return ToContext(**futures)
 

    def process_supercells(self):
        for input_cif_file in self.ctx.structures_used:
            replicate_name = 'replicate.%s' % (input_cif_file)
            if not (replicate_name in self.ctx):
                continue
            self.ctx.structures_used[input_cif_file]['structure_supercell'] = self.ctx[replicate_name].get_outputs(StructureData, link_type=LinkType.RETURN)[0].get_pymatgen()

            
    def generate_configurations(self):
        partial_input_dict = dict(equilibration=self.ctx.equilibration,
                                  temperature=self.ctx.mc_temp,
                                  selection='last',
                                  pick_conf_every=1,
                                  n_rounds=1,
                                  n_conf_target=1,
                                  return_unique=True)
        
        futures = {}
        for input_cif_file in self.ctx.structures_used:
            structure = self.ctx.structures_used[input_cif_file]['structure_original']
          
            if structure.composition.as_dict() == structure.composition.element_composition.as_dict():
                dict_charges = self.ctx.oxidation_states
            else:
                dict_charges = {}
                for key in structure.composition.as_dict():
                    q = re.sub('[^+,-]', '', key)
                    q += re.sub('[^0-9,.]', '', key)
                    if re.sub('[^0-9,.]', '', key) is '':
                        q += '1'
                    charge = float(q)
                    element = re.sub('[+,-]', '', key)
                    element = re.sub('[0-9,.]', '', element)
                    dict_charges[element] = charge
                    
            self.ctx.structures_used[input_cif_file]['dict_charges'] = dict_charges
            partial_input_dict['charges'] = dict_charges
            
            structure_supercell = self.ctx.structures_used[input_cif_file]['structure_supercell']
            structure_supercell.remove_oxidation_states()
            
            self.ctx.structures_used[input_cif_file]['is_partial'] = not structure_supercell.is_ordered
            
            if structure_supercell.is_ordered:
                self.ctx.structures_used[input_cif_file]['structure_ordered'] = structure_supercell
            else:
                futures['mc.%s' % (input_cif_file)] = self.submit(PartialOccupancyWorkChain,
                                                             structure=StructureData(pymatgen=structure_supercell),
                                                             parameters=ParameterData(dict=partial_input_dict),
                                                             seed=Int(self.ctx.rs.randint(2**31 - 1)))
 
        return ToContext(**futures)


    def process_configurations(self):
        for input_cif_file in self.ctx.structures_used:
            mc_name = 'mc.%s' % (input_cif_file)
            if not (mc_name in self.ctx):
                continue
            self.ctx.structures_used[input_cif_file]['structure_ordered'] = self.ctx[mc_name].get_outputs(StructureData, link_type=LinkType.RETURN)[0].get_pymatgen()
            
        for input_cif_file in self.ctx.structures_used:
            self.ctx.structures_used[input_cif_file]['composition_dict'] = \
                        self.ctx.structures_used[input_cif_file]['structure_ordered'].composition.element_composition.as_dict()
            self.ctx.structures_used[input_cif_file]['volume'] = \
                        self.ctx.structures_used[input_cif_file]['structure_ordered'].volume
            self.ctx.structures_used[input_cif_file]['energy'] = None
            self.ctx.structures_used[input_cif_file]['min_key'] = None

            
    def run_calc(self):
        futures = {}
        for input_cif_file in self.ctx.structures_used:
            structure = self.ctx.structures_used[input_cif_file]['structure_ordered']
            parameters = deepcopy(self.ctx.energy_parameters)
            
            for i, el in enumerate(structure.composition.element_composition):
                parameters.get('SYSTEM').setdefault('starting_magnetization(%d)' % (i + 1), 0.1)
            
            ecutrho = 800.0
            ecutwfc = 100.0
            try:
                cutoff_data = get_pseudos(structure=StructureData(pymatgen=structure), 
                                      pseudo_family_name=self.ctx.pseudo_family.value,
                                      return_cutoffs=True)
                ecutrho = 2 * cutoff_data.get('ecutrho')
                ecutwfc = 2 * cutoff_data.get('ecutwfc')
            except:
                pass
                
            parameters.get('SYSTEM').setdefault('ecutrho', ecutrho)
            parameters.get('SYSTEM').setdefault('ecutwfc', ecutwfc)
            
            futures['energy.%s' % (input_cif_file)] = self.submit(
                    WorkflowFactory('quantumespresso.pw.relax'),
                    structure=StructureData(pymatgen=structure),
                    pseudo_family=self.ctx.pseudo_family,
                    max_iterations=Int(1),
                    max_meta_convergence_iterations=Int(1),
                    parameters=ParameterData(dict=parameters),
                    **self.ctx.energy_inputs
                )
            
        return ToContext(**futures)

    
    def process_calc(self):
        for input_cif_file in self.ctx.structures_used:
            calc_name = 'energy.%s' % (input_cif_file)
            if not (calc_name in self.ctx):
                continue
            
            try:
                calc = self.ctx[calc_name].get_outputs(link_type=LinkType.CALL)[0].get_outputs(link_type=LinkType.CALL)[0]
                self.ctx.structures_used[input_cif_file]['energy'] = calc.get_outputs_dict()['output_parameters'].get_dict().get('energy')
            except:
                continue
            
            self.out('structure.%s' % input_cif_file, \
                     StructureData(pymatgen=self.ctx.structures_used[input_cif_file]['structure_ordered']))
            
            properties_dict = {'is_partial': self.ctx.structures_used[input_cif_file]['is_partial'],
                               'dict_charges': self.ctx.structures_used[input_cif_file]['dict_charges'],
                               'volume': self.ctx.structures_used[input_cif_file]['volume'],
                               'composition_dict': self.ctx.structures_used[input_cif_file]['composition_dict'],
                               'energy': self.ctx.structures_used[input_cif_file]['energy']}
            
            self.out('properties.%s' % input_cif_file, ParameterData(dict=properties_dict))
            
        structures_min_energy_out = {}
        for input_cif_file in self.ctx.structures_used:
            min_key = self.__get_min_energy_key(self.ctx.structures_used[input_cif_file]['composition_dict'], self.ctx.structures_used)
            self.ctx.structures_used[input_cif_file]['min_key'] = min_key
            if min_key is not None:
                self.ctx.structures_min_energy[min_key] = self.ctx.structures_used[min_key]
                structures_min_energy_out[min_key] = {'energy': self.ctx.structures_min_energy[min_key]['energy'],
                                                      'composition_dict': self.ctx.structures_min_energy[min_key]['composition_dict']}
                
        self.out('structures_min_energy', ParameterData(dict=structures_min_energy_out))
                
                
    def compute_potentials(self):
        structures_dict = self.ctx.structures_min_energy
        
        main_comp_key = self.__get_min_energy_key(self.ctx.main_composition, self.ctx.structures_min_energy)
        ref_comp_key = self.__get_min_energy_key(self.ctx.ref_composition, self.ctx.structures_min_energy)
        
        main_comp_dict = structures_dict[main_comp_key]['composition_dict']
        main_comp_elements = main_comp_dict.keys()
        num_elements = len(main_comp_elements)
        main_comp_energy = float(structures_dict[main_comp_key]['energy'])

        ref_comp_dict = structures_dict[ref_comp_key]['composition_dict']
        ref_comp_energy = float(structures_dict[ref_comp_key]['energy'])

        keys = structures_dict.keys()
        num_products = num_elements-1
        product_tuples = list(itertools.combinations(keys, num_products))
        if len(product_tuples) > self.ctx.max_num_reactions:
            self.report('Number of product combinations: ' + str(len(product_tuples)))
            return self.exit_codes.ERROR_MAX_NUM_REACTIONS

        potentials_ox_mu = []
        potentials_ox_sig = []
        potentials_red_mu = []
        potentials_red_sig = []
        
        potential_ox = None
        potential_ox_error = None
        products_ox = None
        products_ox_coeffs = None
        products_ox_files = None
        
        potential_red = None
        potential_red_error = None
        products_red = None
        products_red_coeffs = None
        products_red_files = None
        
        decomp_energy = None
        decomp_products = None
        decomp_products_coeffs = None
        decomp_products_files = None
        
        for product_tuple in product_tuples:
            try:
                product_dicts = [structures_dict[product]['composition_dict'] for product in product_tuple]
            except KeyError:
                continue

            product_dicts.append(ref_comp_dict)
            coeffs = self.__expand_stoichiometry(main_comp_dict, product_dicts, self.ctx.det_thr)

            success = True
            if any(coeffs[i] < -self.ctx.coeff_thr for i in range(num_products)):
                success = False

            if all(np.abs(coeffs[i]) < self.ctx.coeff_thr for i in range(len(coeffs))):
                success = False

            is_ox = False
            is_red = False
            if coeffs[num_products] > self.ctx.coeff_thr:
                is_ox = True
            elif coeffs[num_products] < -self.ctx.coeff_thr:
                is_red = True

            if success \
                    and not (is_ox or is_red) \
                    and all([structures_dict[product]['energy'] is not None for product in product_tuple]):
                decomp_energy_tmp = -main_comp_energy
                i = -1
                for product in product_tuple:
                    i += 1
                    decomp_energy_tmp = decomp_energy_tmp + coeffs[i] * structures_dict[product]['energy']
                decomp_energy_tmp = decomp_energy_tmp + coeffs[num_products] * ref_comp_energy
                if decomp_energy_tmp < decomp_energy or decomp_energy is None:
                    decomp_energy = decomp_energy_tmp
                    decomp_products = product_dicts
                    decomp_products_coeffs = coeffs
                    decomp_products_files = product_tuple

            if success \
                    and (is_ox or is_red) \
                    and all([structures_dict[product]['energy'] is not None for product in product_tuple]):
                chem_pot_eq = main_comp_energy
                chem_pot_eq_error_sq = self.ctx.energy_supercell_error**2
                i = -1
                for product in product_tuple:
                    i += 1
                    chem_pot_eq = chem_pot_eq - coeffs[i] * structures_dict[product]['energy']
                    chem_pot_eq_error_sq += (coeffs[i] * self.ctx.energy_supercell_error)**2
                chem_pot_eq = chem_pot_eq / coeffs[num_products] / float(ref_comp_dict[self.ctx.mobile_species])
                chem_pot_eq_error_sq = chem_pot_eq_error_sq / \
                                       ((coeffs[num_products] * float(ref_comp_dict[self.ctx.mobile_species]))**2)
                potential_eq = -(chem_pot_eq - (ref_comp_energy / float(ref_comp_dict[self.ctx.mobile_species])))
                potential_eq_error_sq = chem_pot_eq_error_sq + \
                                        ((self.ctx.energy_supercell_error / float(ref_comp_dict[self.ctx.mobile_species]))**2)
                potential_eq_error = np.sqrt(potential_eq_error_sq)

                if is_ox:
                    potentials_ox_mu.append(potential_eq)
                    potentials_ox_sig.append(potential_eq_error)
                    if potential_eq < potential_ox or potential_ox is None:
                        if potential_eq_error <= self.ctx.error_tol_potential or self.ctx.error_tol_potential is None:
                            potential_ox = potential_eq
                            potential_ox_error = potential_eq_error
                            products_ox = product_dicts
                            products_ox_coeffs = coeffs
                            products_ox_files = product_tuple
                elif is_red:
                    potentials_red_mu.append(potential_eq)
                    potentials_red_sig.append(potential_eq_error)
                    if potential_eq > potential_red or potential_red is None:
                        if potential_eq_error <= self.ctx.error_tol_potential or self.ctx.error_tol_potential is None:
                            potential_red = potential_eq
                            potential_red_error = potential_eq_error
                            products_red = product_dicts
                            products_red_coeffs = coeffs
                            products_red_files = product_tuple

        potentials_all_dict = {}
       
        potentials_all_dict['potentials_ox_mu'] = potentials_ox_mu
        potentials_all_dict['potentials_ox_sig'] = potentials_ox_sig
        potentials_all_dict['potentials_red_mu'] = potentials_red_mu
        potentials_all_dict['potentials_red_sig'] = potentials_red_sig
        
        decomp_relevant_dict = {}
        
        decomp_relevant_dict['main_comp_file'] = main_comp_key
        decomp_relevant_dict['main_comp_dict'] = main_comp_dict
        decomp_relevant_dict['ref_comp_file'] = ref_comp_key
        decomp_relevant_dict['ref_comp_dict'] = ref_comp_dict
        
        decomp_relevant_dict['potential_ox'] = potential_ox
        decomp_relevant_dict['potential_ox_error'] = potential_ox_error
        decomp_relevant_dict['products_ox'] = products_ox
        decomp_relevant_dict['products_ox_coeffs'] = products_ox_coeffs
        decomp_relevant_dict['products_ox_files'] = products_ox_files
        
        decomp_relevant_dict['potential_red'] = potential_red
        decomp_relevant_dict['potential_red_error'] = potential_red_error
        decomp_relevant_dict['products_red'] = products_red
        decomp_relevant_dict['products_red_coeffs'] = products_red_coeffs
        decomp_relevant_dict['products_red_files'] = products_red_files
        
#         decomp_relevant_dict['phase_decomp_energy'] = decomp_energy
#         decomp_relevant_dict['phase_decomp_products'] = decomp_products
#         decomp_relevant_dict['phase_decomp_products_coeffs'] = decomp_products_coeffs
#         decomp_relevant_dict['phase_decomp_products_files'] = decomp_products_files
        
        self.out('potentials_all', ParameterData(dict=potentials_all_dict))
        self.out('decomp_relevant', ParameterData(dict=decomp_relevant_dict))

        
    def __get_min_energy_key(self, composition, structure_dict, eps = 1e-6):
        min_energy = None
        min_key = None
        for key in structure_dict:
            if 'composition_dict' not in structure_dict[key]:
                continue
            if 'energy' not in structure_dict[key]:
                continue
            if structure_dict[key]['energy'] is None:
                continue

            skip = False
            for ele in composition:
                if ele not in structure_dict[key]['composition_dict']:
                    skip = True
            if skip:
                continue

            scales = []
            for ele, stoi in structure_dict[key]['composition_dict'].items():
                if ele not in composition:
                    skip = True
                else:
                    scales.append(composition[ele]/stoi)
            if skip:
                continue

            for scale in scales:
                if np.abs(scale - scales[0]) > eps:
                    skip = True
            if skip:
                continue
            else:
                energy = scales[0] * structure_dict[key]['energy']
                if energy < min_energy or min_energy is None:
                    min_energy = energy
                    min_key = key

        return min_key


    def __expand_stoichiometry(self, main_comp_dict, product_dicts, det_min):
        main_comp_elements = list(main_comp_dict.keys())
        num_elements = len(main_comp_elements)
        num_products = len(product_dicts)

        v_stoi_in = [main_comp_dict[element] for element in main_comp_elements]
        v_stoi_out = np.zeros(num_products)
        h = np.zeros([num_elements, num_products])
        for i in range(num_elements):
            for j in range(num_products):
                h[i][j] = product_dicts[j][main_comp_elements[i]]

        if np.abs(np.linalg.det(h)) >= det_min:
                v_stoi_out = np.matmul(np.linalg.inv(h), v_stoi_in)

        return v_stoi_out
    
