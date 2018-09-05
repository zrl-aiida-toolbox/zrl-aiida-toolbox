import numpy as np

from pymatgen.core.lattice import Lattice
from aiida.orm import DataFactory
from aiida.work.workchain import WorkChain

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
Int = DataFactory('int')


class ShakeWorkChain(WorkChain):
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

        spec.output_namespace('structures', valid_type=StructureData, dynamic=True)

    def validate_inputs(self):
        parameter_dict = self.inputs.parameters.get_dict()

        self.ctx.seed = self.inputs.seed if 'seed' in self.inputs else Int(np.random.randint(2**32 - 1))
        self.ctx.rs = np.random.RandomState(seed=self.ctx.seed.value)

        self.ctx.stdev_atms = float(parameter_dict.get('stdev_atms', 0.01))
        self.ctx.stdev_cell = float(parameter_dict.get('stdev_cell', 0))
        self.ctx.n = int(parameter_dict.get('n', 1))

        self.ctx.structure = self.inputs.structure.get_pymatgen()

    def __with_noise(self):
        new = self.ctx.structure.copy()
        for i, site in enumerate(new):
            d = self.ctx.rs.normal(scale=self.ctx.stdev_atms, size=3)
            new.translate_sites(i, d, frac_coords=False)
            if self.ctx.stdev_cell:
                lattice = Lattice(new.lattice.matrix + np.tril(self.ctx.rs.normal(loc=0,
                                                                                  scale=self.ctx.stdev_cell,
                                                                                  size=9)
                                                               .reshape((3, 3))))
                new.modify_lattice(lattice)
            structure = StructureData(pymatgen=new)
            return structure.uuid, structure

    def shake(self):
        for i in range(self.ctx.n):
            uuid, structure = self.__with_noise()
            self.out('structures.%s' % uuid, structure)
