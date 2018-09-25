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
Int = DataFactory('int')


class PartialOccupancyWorkChain(WorkChain):
    __k_b = 8.61733e-5

    @classmethod
    def define(cls, spec):
        super(WorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('seed', valid_type=Int, required=False)

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

    def validate_inputs(self):
        parameter_dict = self.inputs.parameters.get_dict()
        
        self.ctx.seed = self.inputs.seed if 'seed' in self.inputs else Int(np.random.randint(2**32 - 1))
        self.ctx.rs = np.random.RandomState(seed=self.ctx.seed.value)
        
        self.ctx.charges = {
            key: float(value)
            for key, value in parameter_dict.get('charges', {}).items()
        }        
        self.ctx.structure = self.inputs.structure.get_pymatgen()
        
        self.ctx.partials = []
        for site in self.ctx.structure:
            species = site.species_and_occu.as_dict().items()
            if (len(species) > 1 or species[0][1] < 1) and species not in self.ctx.partials:
                self.ctx.partials.append(species)
        
        self.ctx.vacancy = Specie(parameter_dict.get('vacancy_ion') if 'vacancy_ion' in parameter_dict else 'Lr', 0)
        
        self.ctx.temperature = float(parameter_dict.get('temperature', 1000))
        self.ctx.n_conf_target = int(parameter_dict.get('n_conf_target', 1))
        self.ctx.pick_conf_every = int(parameter_dict.get('pick_conf_every', 25))
        self.ctx.n_rounds = int(parameter_dict.get('n_rounds', self.ctx.n_conf_target + 10))
        
        self.ctx.round = 0
        self.ctx.do_break = False
        
        self.out('seed', Int(self.ctx.seed))
        
    def initialize(self):
        """
        Prepare self.ctx.static, a list of all the sites with static (full) occupancy.
        Prepare self.ctx.sites, a dictionary containing, for each type of partially occupied site, the list of site,
        with either the ion, or the vacancy site.
        :return:
        """
        self.ctx.static = [
            PeriodicSite(Specie(site.species_and_occu.items()[0][0].value),
                         site.coords, site.lattice, True, True)
            for site in self.ctx.structure if site.species_and_occu.as_dict().items() not in self.ctx.partials
        ]

        q, sites = reduce(lambda q, acc: self.__sites_reducer(q, acc), self.ctx.structure, (0, {}))

        self.ctx.sites = dict()

        counts = {}
        for total, species in sorted(sites.keys(), key=lambda x: x[0], reverse=True):
            site = sites.get((total, species))
            new_sites = []
            start = 0
            itr = iter(site)
            try:
                for specie, occupancy_target in chain(species, ((self.ctx.vacancy, 1),)):
                    occupancy = 0
                    while counts.get(specie.value, 0) + 1 < self.ctx.structure.composition.get(specie.value) + 0.5:
                        occupancy_tmp = (1. + len(new_sites) - start) / len(site)
                        self.report((specie.value, occupancy, occupancy_tmp))
                        
                        if np.abs(occupancy_tmp - occupancy_target) > np.abs(occupancy - occupancy_target):
                            break
                        
                        occupancy = occupancy_tmp
                        new_site = next(itr)
                        counts.setdefault(specie.value, 0)
                        counts[specie.value] += 1
                        new_sites.append(PeriodicSite(Specie(specie.value, self.ctx.charges.get(specie.value, 0)),
                                                      new_site.coords, new_site.lattice, True, True))
                    start = len(new_sites)
                while True:
                    new_site = next(itr)
                    new_sites.append(PeriodicSite(self.ctx.vacancy, new_site.coords, new_site.lattice, True, True))
            except StopIteration:
                self.ctx.sites[(total, species)] = new_sites
        self.ctx.energy = [self.__ewald(self.ctx.sites)]

    def do_rounds(self):
        return self.ctx.round < self.ctx.n_rounds and not self.ctx.do_break
    
    def round(self):
        energies = []
        swaps = []
        self.ctx.configurations = []
        
        for i in range(self.ctx.pick_conf_every):
            swaps.append(0)
            for specie in self.ctx.sites.keys():
                energy, swapped = self.__swap(specie)
                if swapped:
                    swaps[-1] += 1
                    self.ctx.configurations = (self.ctx.configurations[- (self.ctx.n_conf_target - 1):]
                                               if self.ctx.n_conf_target > 1
                                               else []) + [deepcopy(self.ctx.sites)]
            energies.append(energy)
        
        self.ctx.round += 1
        
        if np.std(energies) == 0:
            self.ctx.do_break = True
                    
    def finalize(self):
        for configuration in self.ctx.configurations:
            structure = Structure.from_sites(self.ctx.static
                                             + sum([filter(lambda v: v.specie.element.value != self.ctx.vacancy.element.value,
                                                           value) for value in configuration.values()], []))
            structure.sort()
            structure = StructureData(pymatgen=structure)
            self.out('structures.%s' % structure.uuid, structure)
        
    def __ewald(self, sites):
        structure = Structure\
            .from_sites(sum([
                filter(lambda v: v.specie.element.value != self.ctx.vacancy.element.value, value)
                for value in sites.values()
            ], []))

        return EwaldSummation(structure).total_energy

    def __sites_reducer(self, (q, acc), cur):
        species = tuple(cur.species_and_occu.items())
        if len(species) > 1 or species[0][1] < 1:
            acc.setdefault((sum([x[1] for x in species]), species), [])
            acc.get((sum([x[1] for x in species]), species)).append(cur)
        else:
            q += self.ctx.charges.get(species[0][0].value, 0)
        return q, acc

    def __swap(self, species):
        new_sites = deepcopy(self.ctx.sites)

        i = self.ctx.rs.randint(len(new_sites[species]))
        isite = self.ctx.sites[species][i]
        ispecie = isite.specie

        J = [j for j, site in enumerate(new_sites[species]) if site.specie != ispecie]

        if len(J) == 0:
            return self.ctx.energy[-1], False

        j = self.ctx.rs.randint(len(J))

        jsite = self.ctx.sites[species][J[j]]
        jspecie = jsite.specie

        new_sites[species][i] = PeriodicSite(jspecie, isite.coords, isite.lattice, True, True)
        new_sites[species][J[j]] = PeriodicSite(ispecie, jsite.coords, jsite.lattice, True, True)

        energy = self.__ewald(new_sites)

        if np.exp(np.min([500, -(energy - self.ctx.energy[-1]) / (self.__k_b * self.ctx.temperature)])) > self.ctx.rs.rand():
            self.ctx.sites = new_sites
            return energy, True
        return self.ctx.energy[-1], False
