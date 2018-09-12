import numpy as np

from aiida.orm import DataFactory
from aiida.work.workchain import WorkChain, if_

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')


class ReplicateWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(WorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData)

        spec.outline(
            cls.validate_inputs,
            if_(cls.abc_not_defined)(
                cls.calculate_factors
            ),
            cls.replicate
        )

        spec.output('structure', valid_type=StructureData)

    def validate_inputs(self):
        parameter_dict = self.inputs.parameters.get_dict()
        self.ctx.a = int(parameter_dict.get('a')) if 'a' in parameter_dict else None
        self.ctx.b = int(parameter_dict.get('b')) if 'b' in parameter_dict else None
        self.ctx.c = int(parameter_dict.get('c')) if 'c' in parameter_dict else None
        self.ctx.max_electrons = int(parameter_dict.get('max_electrons')) if 'max_electrons' in parameter_dict else None
        self.ctx.val_electrons = {
            key: int(value) for key, value in parameter_dict.get('val_electrons', {}).items()
        }
        self.ctx.max_volume = float(parameter_dict.get('max_volume')) if 'max_volume' in parameter_dict else None
        self.ctx.structure = self.inputs.structure.get_pymatgen()

        assert self.ctx.a and self.ctx.b and self.ctx.c \
               or (self.ctx.max_electrons and len(self.ctx.val_electrons)) \
               or self.ctx.max_volume, \
            'You must either provide values for `a`, `b` and `c` or for `max_electrons` in the parameters dictionary.'

    def abc_not_defined(self):
        return self.ctx.a is None and self.ctx.b is None and self.ctx.c is None

    def calculate_factors(self):
        volume = self.ctx.structure.volume

        if self.ctx.max_electrons:
            electrons = np.floor(sum(n * self.ctx.val_electrons.get(symbol)
                                     for symbol, n in self.ctx.structure.composition.as_dict().items()))
            max_volume = volume * self.ctx.max_electrons / electrons
        else:
            max_volume = self.ctx.max_volume

        max_edge = np.power(max_volume, 1. / 3)
        replicas = np.floor(max_edge / np.diag(self.ctx.structure.lattice.matrix)).astype(int)
        replicas[np.where(replicas == 0)] = 1

        self.ctx.a, self.ctx.b, self.ctx.c = replicas

    def replicate(self):
        self.out('structure',
                 StructureData(pymatgen=self.ctx.structure * np.array([self.ctx.a, self.ctx.b, self.ctx.c], dtype=int)))
