import numpy as np
from enum import Enum, unique

import scipy.stats as stats
from itertools import count

from pymatgen.core.lattice import Lattice
from aiida.orm import DataFactory
from aiida.work.workchain import WorkChain

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
Int = DataFactory('int')


class ShakeWorkChain(WorkChain):
    
    @unique
    class Distribution(Enum):
        MAXWELL = 'maxwell'
        # scale = self.ctx.stdev_atms / np.sqrt((3 * np.pi - 8) / np.pi)
        
        NORM = 'norm'
        # scale = self.ctx.stdev_atms ** 2
        
        EXPON = 'expon'
        # scale = self.ctx.stdev_atms
    
    @classmethod
    def define(cls, spec):
        super(WorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('seed', valid_type=Int, required=False)

        spec.outline(
            cls.validate_inputs,
            cls.shake
        )

        spec.output('seed', valid_type=Int)
        spec.output_namespace('structures', valid_type=StructureData, dynamic=True)
        
        spec.exit_code(1, 'ERROR_INVALID_DISTRIBUTION', 'Invalid or non-supported distribution provided.')

    def validate_inputs(self):
        parameter_dict = self.inputs.parameters.get_dict()

        self.ctx.seed = self.inputs.seed if 'seed' in self.inputs else Int(np.random.randint(2**31 - 1))
        self.ctx.rs = np.random.RandomState(seed=self.ctx.seed.value)

        try:
            self.ctx.distribution = self.Distribution(parameter_dict.get('distribution', 'norm'))
        except ValueError as e:
            return self.exit_codes.ERROR_INVALID_DISTRIBUTION
            
        self.ctx.stdev_atms = float(parameter_dict.get('stdev_atms', 0))        
        self.ctx.stdev_cell = float(parameter_dict.get('stdev_cell', 0))
        
        self.ctx.n = int(parameter_dict.get('n', 1))

        self.ctx.structure = self.inputs.structure.get_pymatgen()
        self.out('seed', self.ctx.seed)

    def __with_noise(self):
        new = self.ctx.structure.copy()
        
        directions = self.ctx.rs.rand(new.num_sites, 3) - 0.5
        directions /= np.linalg.norm(directions, axis=-1)[:, None]
        
        distribution = getattr(stats, self.ctx.distribution.value)
        
        for i, site, direction in zip(count(0), new, directions):
            norm = distribution.rvs(size=1, loc=0, scale=self.scale(), random_state=self.ctx.rs)
            d = norm * direction
            new.translate_sites(i, d, frac_coords=False)
            
        if self.ctx.stdev_cell:
            mask = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1]).reshape(3, 3)
            delta = \
                self.ctx.rs.normal(loc=0, scale=self.ctx.stdev_cell, size=9)\
                    .reshape(3, 3) * mask
            
            lattice = Lattice(new.lattice.matrix + delta)
            new.modify_lattice(lattice)
        
        structure = StructureData(pymatgen=new)
        
        return structure.uuid, structure

    def scale(self):
        if self.ctx.distribution == self.Distribution.MAXWELL:
            return self.ctx.stdev_atms / np.sqrt((3 * np.pi - 8) / np.pi)
        if self.ctx.distribution == self.Distribution.NORM:
            return self.ctx.stdev_atms ** 2
        if self.ctx.distribution == self.Distribution.EXPON:
            return self.ctx.stdev_atms
    
    def shake(self):
        for i in range(self.ctx.n):
            uuid, structure = self.__with_noise()
            self.out('structures.%s' % uuid, structure)
