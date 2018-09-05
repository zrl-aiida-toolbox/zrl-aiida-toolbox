from copy import deepcopy
from itertools import chain

import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.sites import PeriodicSite, Specie
from pymatgen.core.structure import Structure

from aiida.orm import DataFactory
from aiida.work.workchain import WorkChain

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
            cls.generate
        )

        spec.output_namespace('structures', valid_type=StructureData, dynamic=True)
        spec.output('seed', valid_type=Int)

    def validate_inputs(self):
        parameter_dict = self.inputs.parameters.get_dict()

        self.ctx.seed = self.inputs.seed if 'seed' in self.inputs else Int(np.random.randint(2**32 - 1))
        self.out('seed', self.ctx.seed)
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
        self.ctx.max_rounds = int(parameter_dict.get('max_rounds', 50))
        self.ctx.max_configurations = int(parameter_dict.get('max_configurations', 1))

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

        for total, species in sorted(sites.keys(), key=lambda x: x[0], reverse=True):
            site = sites.get((total, species))
            new_sites = []
            start = 0
            itr = iter(site)
            try:
                for specie, occup in chain(species, ((self.ctx.vacancy, 1),)):
                    while q != 0 and (1. * len(new_sites) - start) / len(site) < occup:
                        new_site = next(itr)
                        q += self.ctx.charges.get(specie.value, 0)
                        new_sites.append(PeriodicSite(Specie(specie.value, self.ctx.charges.get(specie.value, 0)),
                                                      new_site.coords, new_site.lattice, True, True))
                    start = len(new_sites)
                while True:
                    new_site = next(itr)
                    new_sites.append(PeriodicSite(self.ctx.vacancy, new_site.coords, new_site.lattice, True, True))
            except StopIteration:
                pass

            self.ctx.sites[(total, species)] = new_sites

        self.ctx.energy = [self.__ewald(self.ctx.sites)]

    def generate(self):
        swaps = []
        sites = []
        try:
            for i in range(self.ctx.max_rounds):
                swaps.append(0)
                for j in range(25):
                    for specie in self.ctx.sites.keys():
                        energy, swapped = self.__swap(specie, T=1800)
                        if swapped:
                            swaps[-1] += 1
                            sites = sites[-(self.ctx.max_configurations - 1):] + [deepcopy(self.ctx.sites)]
                        self.ctx.energy.append(energy)
                if sum(swaps[-10:]) == 0:
                    break
        except KeyboardInterrupt:
            pass

        self.ctx.configurations = [
            Structure.from_sites(self.ctx.static
                                 + sum([filter(lambda v: v.specie.element.value != self.ctx.vacancy.element.value,
                                               value) for value in site.values()], []))
            for site in sites
        ]

        for configuration in self.ctx.configurations:
            configuration.sort()
            structure = StructureData(pymatgen=configuration)
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

    def __swap(self, species, T=300):
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

        if np.exp(np.min([500, -(energy - self.ctx.energy[-1]) / (self.__k_b * T)])) > self.ctx.rs.rand():
            self.ctx.sites = new_sites
            return energy, True
        return self.ctx.energy[-1], False
