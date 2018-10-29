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


# This workchains changes the stoichiometry of a specific species inside a given structure by filling/creating vacancies. The change of the total number of atoms of this species is given as input referring to the (super)cell specified in the structure object. If the species occupies several sites, atoms are removed from those sites with lowest occupation and added to those sites with highest occupation.

# @workfunction
# def ChangeStoichiometryWorkFunction():

class ChangeStoichiometryWorkChain(WorkChain):
    
    @classmethod
    def define(cls, spec):    
        super(ChangeStoichiometryWorkChain, cls).define(spec)
        
        spec.input('structure', valid_type=StructureData, required=True)
        spec.input('species', valid_type=Str, required=True)
        spec.input('delta_N', valid_type=Int, required=True)
        spec.input('distribution', valid_type=Str, default=Str('equal'))

        spec.outline(
            cls.check_inputs,
            if_(cls.distribution_is_equal)(
                cls.change_stoichiometry_equal,
            ).elif_(cls.distribution_is_aufbau)(
                cls.change_stoichiometry_aufbau,
            ).else_(
                cls.no_distribution,
            ),
            cls.change_stoichiometry,
            cls.set_output
        )
        
        spec.output_namespace('test', valid_type=Float, dynamic=True)

        spec.exit_code(1, 'ERROR_MISSING_DISTRIBUTION', 'No valid distibution for stoichiometry change provided.')

        
    def check_inputs(self):
        pass

    
    def distribution_is_equal(self):
        return self.inputs.distribution.value == 'equal'

    
    def distribution_is_aufbau(self):
        return self.inputs.distribution.value == 'aufbau'

    
    def no_distribution(self):
        return self.exit_codes.ERROR_MISSING_DISTRIBUTION
    
    
    def change_stoichiometry_equal(self):
        pass
 

    def change_stoichiometry_aufbau(self):
        structure = self.inputs.structure
        species = self.inputs.species.value
        delta_N = self.inputs.deltaN.value
        delta_weight_equal = 0.01
        error_tol_occ = 1e-4

        if (abs(delta_N) <= error_tol_occ):
            print('ERROR')

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
                    print('ERROR')

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
                        print('ERROR')

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
                            print('ERROR')

                        change = min([atms_to_add_per_site, (1.0-weight_total)])
                        atms_to_add = atms_to_add - change
                        num_sites -= 1

                        weights_out = list(kind.weights)
                        weights_out[species_index] += change
                        struct_out.append_atom(position=site.position, symbols=kind.symbols, weights=weights_out)

        if (abs(atms_to_add) > error_tol_occ):
            print('ERROR')
            print(atms_to_add)

        for i in fixed_sites:
                site = structure.sites[i]
                kind = structure.get_kind(site.kind_name)
                struct_out.append_atom(position=site.position, symbols=kind.symbols, weights=kind.weights)

        struct_out.__dict__

    
    def set_output(self):
        pass


    