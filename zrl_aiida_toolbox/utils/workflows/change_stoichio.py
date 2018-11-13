from aiida.work.workchain import WorkChain, ToContext, if_, while_, return_
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.orm import Code, LinkType, DataFactory, WorkflowFactory
from aiida.work.launch import run
from aiida.work.run import submit

from pymatgen.core.structure import Structure

Str = DataFactory('str')
Float = DataFactory('float')
Int = DataFactory('int')

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')


# This workchains changes the stoichiometry of a specific species inside a given structure by filling/creating vacancies. The change of the total number of atoms of this species is given as input referring to the (super)cell specified in the structure object. If the species occupies several sites, the following distribution schemes are available:
# 'aufbau': Atoms are removed from those sites with lowest occupation and added to those sites with highest occupation.
# 'equal_scale': All site occupancies are scaled with a common factor


class ChangeStoichiometryWorkChain(WorkChain):
    
    @classmethod
    def define(cls, spec):    
        super(ChangeStoichiometryWorkChain, cls).define(spec)
        
        spec.input('structure', valid_type=StructureData, required=True)
        spec.input('species', valid_type=Str, required=True)
        spec.input('delta_N', valid_type=Float, required=True)
        spec.input('distribution', valid_type=Str, default=Str('equal_scale'))
        spec.input('delta_weight_equal', valid_type=Float, default=Float(0.01))
        spec.input('error_tol_occ', valid_type=Float, default=Float(1e-4))

        spec.outline(
            cls.check_inputs,
            if_(cls.distribution_is_equal_scale)(
                cls.change_stoichiometry_equal_scale,
            ).elif_(cls.distribution_is_aufbau)(
                cls.change_stoichiometry_aufbau,
            ).else_(
                cls.no_distribution,
            ),
            cls.set_output
        )
        
        spec.output('structure_changed', valid_type=StructureData)

        spec.exit_code(1, 'ERROR_INPUT', 'Unreasonable input parameters.')
        spec.exit_code(2, 'ERROR_MISSING_DISTRIBUTION', 'No valid distibution for stoichiometry change provided.')
        spec.exit_code(3, 'ERROR_PROCESS', 'Could not generate requested structure: Unreasonable occupancies.')
        spec.exit_code(4, 'ERROR_ATOMS_LEFT', 'Requested change in stoichiometry exceeds amount available in cell.')

        
    def check_inputs(self):
        if (abs(self.inputs.delta_N.value) <= self.inputs.error_tol_occ.value) \
            or (self.inputs.delta_weight_equal.value < 0.0) \
            or (self.inputs.error_tol_occ.value < 0.0):
            return self.exit_codes.ERROR_INPUT

    
    def distribution_is_equal_scale(self):
        return self.inputs.distribution.value == 'equal_scale'

    
    def distribution_is_aufbau(self):
        return self.inputs.distribution.value == 'aufbau'

    
    def no_distribution(self):
        print('ERROR: No valid distribution given.')
        return self.exit_codes.ERROR_MISSING_DISTRIBUTION
    
    
    def change_stoichiometry_equal_scale(self):
        structure = self.inputs.structure
        species = self.inputs.species.value
        delta_N = self.inputs.delta_N.value
        delta_weight_equal = self.inputs.delta_weight_equal.value
        error_tol_occ = self.inputs.error_tol_occ.value
        
        fixed_sites = []
        species_sites = {}
        weight_species_total = 0.0
        for i in range(len(structure.sites)):
            site = structure.sites[i]
            kind = structure.get_kind(site.kind_name)

            weight_species = 0.0
            weight_total = 0.0
            species_count = 0
            for j in range(len(kind.symbols)):
                weight_total += kind.weights[j]
                if (kind.symbols[j] == species):
                    species_count += 1
                    species_index = j
                    weight_species += kind.weights[j]

            if (species_count > 1):
                    self.exit_codes.ERROR_PROCESS

            if (species_count == 1):
                change = 0.0
                change_fixed = False
                species_sites[i] = {'site_index' : i, 'species_index' : species_index, 
                                    'weight_species' : weight_species, 'weight_total' : weight_total, 
                                    'change' : change, 'change_fixed' : change_fixed}
            else:
                fixed_sites.append(i)

            weight_species_total += weight_species

        struct_out = StructureData(cell=structure.cell)

        atms_to_add = delta_N
        atms_to_distribute = delta_N
        weight_species_remain = weight_species_total
        update_changes = True
        while update_changes:
            update_changes = False
            for i in species_sites.keys():
                if not species_sites[i]['change_fixed']:
                    weight_species = species_sites[i]['weight_species']
                    weight_total = species_sites[i]['weight_total']
                    change = atms_to_distribute * weight_species / weight_species_remain
                    if (change > (1.0-weight_total)):
                        change = 1.0-weight_total
                        species_sites[i]['change_fixed'] = True
                        update_changes = True
                        atms_to_distribute -= change
                        weight_species_remain -= weight_species
                    if (change < -weight_species):
                        change = -weight_species
                        species_sites[i]['change_fixed'] = True
                        update_changes = True
                        atms_to_distribute -= change
                        weight_species_remain -= weight_species

                    species_sites[i]['change'] = change

        for i in species_sites.keys():
            site_index = species_sites[i]['site_index']
            species_index = species_sites[i]['species_index']
            site = structure.sites[site_index]
            kind = structure.get_kind(site.kind_name)

            change = species_sites[i]['change']
            atms_to_add = atms_to_add - change

            weight_total = species_sites[i]['weight_total']
            weights_out = list(kind.weights)
            weights_out[species_index] += change
            weight_total_out = weight_total + change
            if (weights_out[species_index] < 0.0 - error_tol_occ) or (weight_total_out > 1.0 + error_tol_occ):
                self.exit_codes.ERROR_PROCESS
            if (weights_out[species_index] < 0.0):
                weights_out[species_index] = 0.0
            if (weight_total_out > 0.0 + error_tol_occ):
                struct_out.append_atom(position=site.position, symbols=kind.symbols, weights=weights_out)

        if (abs(atms_to_add) > error_tol_occ):
            return self.exit_codes.ERROR_ATOMS_LEFT

        for i in fixed_sites:
                site = structure.sites[i]
                kind = structure.get_kind(site.kind_name)
                struct_out.append_atom(position=site.position, symbols=kind.symbols, weights=kind.weights)

        self.ctx.struct_out = struct_out
 

    def change_stoichiometry_aufbau(self):
        structure = self.inputs.structure
        species = self.inputs.species.value
        delta_N = self.inputs.delta_N.value
        delta_weight_equal = self.inputs.delta_weight_equal.value
        error_tol_occ = self.inputs.error_tol_occ.value

        positive_change = True
        if (delta_N < 0.0):
            positive_change = False

        fixed_sites = []
        species_weight_sites = {}
        for i in range(len(structure.sites)):
            site = structure.sites[i]
            kind = structure.get_kind(site.kind_name)

            weight_species = 0.0
            weight_total = 0.0
            species_count = 0
            for j in range(len(kind.symbols)):
                weight_total += kind.weights[j]
                if (kind.symbols[j] == species):
                    species_count += 1
                    species_index = j
                    weight_species += kind.weights[j]

            if (species_count > 1):
                    return self.exit_codes.ERROR_PROCESS

            if (species_count == 1):
                if (weight_species not in species_weight_sites.keys()):
                    species_weight_sites[weight_species] = []
                species_weight_sites[weight_species].append([i, species_index, weight_species, weight_total])
            else:
                fixed_sites.append(i)

        species_weights = species_weight_sites.keys()
        species_weights.sort(reverse=positive_change)
        recheck = True
        while recheck:
            recheck = False
            for i in range(1, len(species_weights)): 
                if (abs(species_weights[i] - species_weights[i-1]) <= delta_weight_equal):
                    if (len(species_weight_sites[species_weights[i]]) > 0):
                        recheck = True
                    species_weight_sites[species_weights[i-1]] += species_weight_sites[species_weights[i]]
                    species_weight_sites[species_weights[i]] = []

        struct_out = StructureData(cell=structure.cell)

        atms_to_add = delta_N
        for weight in species_weights:
            num_sites = len(species_weight_sites[weight])
            weight_total_dict = {}
            for i in species_weight_sites[weight]:
                site_index = i[0]
                species_index = i[1]
                weight_species = i[2]
                weight_total = i[3]
                if (weight_total not in weight_total_dict.keys()):
                    weight_total_dict[weight_total] = []
                weight_total_dict[weight_total].append([site_index, species_index, weight_species, weight_total])

                if not positive_change:
                    if (num_sites > 0):
                        atms_to_add_per_site = atms_to_add / num_sites
                    else:
                        atms_to_add_per_site = 0

                    site = structure.sites[site_index]
                    kind = structure.get_kind(site.kind_name)

                    if (atms_to_add_per_site > 0.0 + error_tol_occ) or (weight_species < 0.0 - error_tol_occ):
                        return self.exit_codes.ERROR_PROCESS

                    change = max([atms_to_add_per_site, -weight_species])
                    atms_to_add = atms_to_add - change
                    num_sites -= 1

                    weights_out = list(kind.weights)
                    weights_out[species_index] += change
                    weight_total_out = weight_total + change
                    if (weights_out[species_index] < 0.0):
                        weights_out[species_index] = 0.0
                    if (weight_total_out > 0.0 + error_tol_occ):
                        struct_out.append_atom(position=site.position, symbols=kind.symbols, weights=weights_out)

            if positive_change:
                weight_total_list = weight_total_dict.keys()
                weight_total_list.sort(reverse=True)

                for weight_total in weight_total_list:
                    for i in weight_total_dict[weight_total]:
                        if (num_sites > 0):
                            atms_to_add_per_site = atms_to_add / num_sites
                        else:
                            atms_to_add_per_site = 0

                        site_index = i[0]
                        species_index = i[1]
                        weight_species = i[2]
                        weight_total = i[3]
                        site = structure.sites[site_index]
                        kind = structure.get_kind(site.kind_name)

                        if (weight_total > 1.0 + error_tol_occ):
                            return self.exit_codes.ERROR_PROCESS

                        change = min([atms_to_add_per_site, (1.0-weight_total)])
                        atms_to_add = atms_to_add - change
                        num_sites -= 1

                        weights_out = list(kind.weights)
                        weights_out[species_index] += change
                        struct_out.append_atom(position=site.position, symbols=kind.symbols, weights=weights_out)

        if (abs(atms_to_add) > error_tol_occ):
            return self.exit_codes.ERROR_ATOMS_LEFT

        for i in fixed_sites:
                site = structure.sites[i]
                kind = structure.get_kind(site.kind_name)
                struct_out.append_atom(position=site.position, symbols=kind.symbols, weights=kind.weights)

        self.ctx.struct_out = struct_out
        
        
    def set_output(self):
        self.out('structure_changed',  self.ctx.struct_out)


    