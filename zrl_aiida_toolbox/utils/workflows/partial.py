from copy import deepcopy
from itertools import chain

import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.sites import PeriodicSite, Specie
from pymatgen.core.structure import Structure

from aiida.orm import DataFactory
from aiida.work.workchain import WorkChain, while_

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
Int = DataFactory('int')
Bool = DataFactory('bool')


class PartialOccupancyWorkChain(WorkChain):
    __k_b = 8.61733e-5

    @classmethod
    def define(cls, spec):
        super(WorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('seed', valid_type=Int, required=False)
        spec.input('verbose', valid_type=Bool, default=Bool(False))
        spec.outline(
            cls.validate_inputs,
            cls.initialize,
            while_(cls.do_rounds)(
                cls.round
            ),
            cls.finalize
        )

        spec.output_namespace('structures', valid_type=StructureData, dynamic=True)
        spec.output('seed', valid_type=Int)
        spec.output('energy', valid_type=ArrayData)

    def validate_inputs(self):
        if self.inputs.verbose:
            self.report('Validating inputs.')
            
        parameter_dict = self.inputs.parameters.get_dict()
        
        self.ctx.seed = self.inputs.seed if 'seed' in self.inputs else Int(np.random.randint(2**31 - 1))
        self.ctx.rs = np.random.RandomState(seed=self.ctx.seed.value)
        
        self.ctx.charges = {
            key: float(value)
            for key, value in parameter_dict.get('charges', {}).items()
        }
        
        self.ctx.structure = self.inputs.structure.get_pymatgen()
        
        for element in self.ctx.structure.composition:
            composition = self.ctx.structure.composition.get(element)
            if composition != np.floor(composition):
                self.report('Warning: the input structure has a non integer composition for element `%s`, you '
                            'might be loosing an atom.' % element)
        
        self.ctx.vacancy = Specie(parameter_dict.get('vacancy_ion') if 'vacancy_ion' in parameter_dict else 'Lr', 0)
        
        self.ctx.temperature = float(parameter_dict.get('temperature', 1000))
        self.ctx.n_conf_target = int(parameter_dict.get('n_conf_target', 1))
        self.ctx.pick_conf_every = int(parameter_dict.get('pick_conf_every', 100))
        self.ctx.equilibration = int(parameter_dict.get('equilibration', 10))
        self.ctx.n_rounds = int(parameter_dict.get('n_rounds', 
                                                   self.ctx.equilibration 
                                                   + self.ctx.n_conf_target 
                                                   + 10))
        self.ctx.unique = bool(parameter_dict.get('return_unique', False))
        self.ctx.selection = dict([('reservoir sampling', 0), ('last', 1)])\
            .get(parameter_dict.get('selection', 'reservoir sampling'), 0)
        
        self.ctx.round = 0
        self.ctx.do_break = 10
        
        self.out('seed', Int(self.ctx.seed))        
        
    def initialize(self):
        """
        Prepare self.ctx.static, a list of all the sites with static (full) occupancy.
        Prepare self.ctx.sites, a dictionary containing, for each type of partially occupied site, the list of site,
        with either the ion, or the vacancy site.
        :return:
        """            
        
        
        fixed_counts = {}
        partial_counts = {}
        
        self.ctx.static = []
        self.ctx.partial = {}
        self.ctx.select = []
        
        # We collect the sites into their composition and calculate the theoretical number of occupied sites
        for site in self.ctx.structure:
            if self.__is_partial(site):
                self.ctx.partial.setdefault(site.species_and_occu, [[]])
                self.ctx.partial.get(site.species_and_occu)[0].append(site)
                
                partial_counts.setdefault(site.species_and_occu, [[0, 0] for s in site.species_and_occu])
                
                for i, element in enumerate(site.species_and_occu):
                    partial_counts[site.species_and_occu][i][0] += site.species_and_occu.get(element)
                    partial_counts[site.species_and_occu][i][1] += site.species_and_occu.get(element)
            else:
                self.ctx.static.append(PeriodicSite(site.specie, site.coords, site.lattice, True, True))
                fixed_counts.setdefault(site.specie, 0)
                fixed_counts[site.specie] += 1
        
        # If all sites are static, then no need to do anything.
        if len(self.ctx.static) == len(self.ctx.structure):
            self.ctx.do_break = 0
            self.out('structures.%s' % self.inputs.structure.uuid, self.inputs.structure)
            return
        
        # We compile the number of occupied site for each partial composition while not going over the theoretical number
        for comp in partial_counts:
            self.ctx.rs.shuffle(self.ctx.partial.get(comp)[0])
            for i, sp in enumerate(comp):
                partial_counts[comp][i][0] = np.floor(partial_counts[comp][i][0])
        
        # Calculation of the departure from the composition. 
        error = {
            el: self.ctx.structure.composition.get(el) - fixed_counts.get(el, 0)
            for el in self.ctx.structure.composition
        }

        for comp in partial_counts:
            for i, sp in enumerate(comp):
                error[sp] -= partial_counts.get(comp)[i][0]

        # Adding ions to sites with the highest departure from theoretical number as long as the error
        # is greater than 0.5.
        for element in error:
            while error[element] > 0.5:
                if error[element] > 0:
                    max_error = (None, 0)
                    for i, comp in enumerate(partial_counts):
                        if element in comp:
                            for j, sp in enumerate(comp):
                                if sp == element:
                                    err = (partial_counts.get(comp)[j][1] - partial_counts.get(comp)[j][0]) ** 2
                                    if err > max_error[1]:
                                        max_error = ((comp, j), err)
                    partial_counts.get(max_error[0][0])[max_error[0][1]][0] += 1
                    error[element] -= 1
        
        self.ctx.configurations = tuple()
        self.ctx.configuration_hashes = tuple()
        # self.ctx.sites = dict()
        
        for comp in partial_counts:
            tmp = np.math.factorial(len(self.ctx.partial.get(comp)[-1]))
            for i, sp in enumerate(comp):
                tmp /= np.math.factorial(int(partial_counts.get(comp)[i][0]))
                for _ in range(int(partial_counts.get(comp)[i][0])):
                    site = self.ctx.partial.get(comp)[-1].pop(0)
                    self.ctx.partial.get(comp).insert(0, PeriodicSite(Specie(sp, self.ctx.charges.get(sp.value, 0)), 
                                                                      site.coords, site.lattice, True, True))
            leftovers = self.ctx.partial.get(comp).pop()
            tmp /= np.math.factorial(len(leftovers))
            for site in leftovers:
                self.ctx.partial.get(comp).insert(0, PeriodicSite(self.ctx.vacancy, 
                                                                  site.coords, site.lattice, True, True))
            for _ in range(np.ceil(np.log10(tmp)).astype(int)):
                self.ctx.select.append(comp)
                                
        self.ctx.idxes = [idx for idx in range(len(self.ctx.select))]
        self.ctx.sites = self.ctx.partial
        del self.ctx.partial
        
        self.ctx.energy = [self.__ewald(self.ctx.sites)]
        
        if self.inputs.verbose:
            self.report('Starting structure: E = %f' % self.ctx.energy[-1])

    def do_rounds(self):
        return self.ctx.round < self.ctx.n_rounds + self.ctx.equilibration and self.ctx.do_break > 0
    
    def round(self):
        swaps = 0
        
        for i in range(self.ctx.pick_conf_every):
            new_sites = self.ctx.sites
            for _ in range(1):
                new_sites = self.__swap(new_sites)
                                    
            energy, swapped = self.__keep(new_sites)
            swaps += swapped
        
        self.ctx.energy.append(energy)
        
        self.ctx.round += 1
        if self.ctx.round > self.ctx.equilibration:
            # Implementation of a reservoir sampling selection after the equilibration steps
            # All structures generated after the equilibration steps have the same probability
            # of being retured.
            structure = self.__get_structure()
            hash = structure.get_hash()
            if len(self.ctx.configurations) < self.ctx.n_conf_target:
                if not self.ctx.unique or hash not in self.ctx.configuration_hashes:
                    self.ctx.configurations += (structure.get_pymatgen(), )
                    self.ctx.configuration_hashes += (hash, )
            else:
                if not self.ctx.unique or hash not in self.ctx.configuration_hashes:
                    if self.ctx.selection == 0:
                        r = self.ctx.rs.randint(2**31 - 1) % (self.ctx.round - self.ctx.equilibration)
                        keep = r < self.ctx.n_conf_target
                        if keep:
                            self.ctx.configurations = self.ctx.configurations[:r] \
                                + (structure.get_pymatgen(), ) \
                                + self.ctx.configurations[r + 1:]
                            self.ctx.configuration_hashes = self.ctx.configuration_hashes[:r] \
                                + (hash, ) \
                                + self.ctx.configuration_hashes[r + 1:]
                    elif self.ctx.selection == 1:
                        self.ctx.configurations = (structure.get_pymatgen(), ) + self.ctx.configurations[-self.ctx.n_conf_target + 1:]
                        self.ctx.configuration_hashes = (hash, ) + self.ctx.configuration_hashes[-self.ctx.n_conf_target + 1:]

        if self.inputs.verbose:
            self.report('Round %4d: E = %f (%d swaps)' % (self.ctx.round, self.ctx.energy[-1], swaps))
                    
    def finalize(self):
        if 'configurations' in self.ctx:
            for structure in self.ctx.configurations:
                structure = StructureData(pymatgen=structure)
                self.out('structures.%s' % structure.uuid, structure) 
            energy = ArrayData()
            energy.set_array('energy', np.array(self.ctx.energy))
            self.out('energy', energy)
    
    def __ewald(self, sites):
        structure = Structure\
            .from_sites(sum([
                list(filter(lambda v: v.specie.element.value != self.ctx.vacancy.element.value, value))
                for value in sites.values()
            ], []))

        return EwaldSummation(structure).total_energy
    
    def __is_partial(self, site):
        for element in site.species_and_occu:
            if site.species_and_occu.get(element) == 1:
                return False    
        return True

    def __swap(self, sites):
        
        idx = self.ctx.rs.choice(self.ctx.idxes)
        species = self.ctx.select[idx]
        
        new_sites = deepcopy(sites)

        i = self.ctx.rs.randint(len(new_sites[species]))
        isite = sites[species][i]
        ispecie = isite.specie
        
        J = [j for j, site in enumerate(new_sites[species]) if site.specie != ispecie]

        # if len(J) == 0:
        #     return self.ctx.energy[-1], False
        
        j = self.ctx.rs.randint(len(J))
        jsite = sites[species][J[j]]
        jspecie = jsite.specie
        
        new_sites[species][i] = PeriodicSite(jspecie, isite.coords, isite.lattice, True, True)
        new_sites[species][J[j]] = PeriodicSite(ispecie, jsite.coords, jsite.lattice, True, True)

        return new_sites
    
    def __keep(self, new_sites):
        energy = self.__ewald(new_sites)

        exp = np.exp(np.min([500, -(energy - self.ctx.energy[-1]) / (self.__k_b * self.ctx.temperature)]))
        r = self.ctx.rs.rand()
        if exp > r:
            self.ctx.sites = new_sites
            return energy, True
        return self.ctx.energy[-1], False
    
    def __get_structure(self):
        structure = Structure.from_sites(self.ctx.static
                                         + sum([[v
                                                 for v in value
                                                 if v.specie.element.value != self.ctx.vacancy.element.value]
                                                for value in self.ctx.sites.values()], []))
        structure.sort()
        return StructureData(pymatgen=structure)
                
