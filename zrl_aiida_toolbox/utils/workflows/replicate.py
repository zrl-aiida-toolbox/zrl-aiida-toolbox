import numpy as np

from aiida.orm import DataFactory
from aiida.work.workchain import WorkChain, if_

StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
Bool = DataFactory('bool')


class ReplicateWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(WorkChain, cls).define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('verbose', valid_type=Bool, default=Bool(False))

        spec.outline(
            cls.validate_inputs,
            if_(cls.abc_not_defined)(
                cls.determine_factors
            ),
            cls.replicate
        )

        spec.output('structure', valid_type=StructureData)

    def validate_inputs(self):
        parameter_dict = self.inputs.parameters.get_dict()
        self.ctx.a = int(parameter_dict.get('a')) if 'a' in parameter_dict else None
        self.ctx.b = int(parameter_dict.get('b')) if 'b' in parameter_dict else None
        self.ctx.c = int(parameter_dict.get('c')) if 'c' in parameter_dict else None
        self.ctx.val_electrons = {
            key: int(value) for key, value in parameter_dict.get('val_electrons', {}).items()
        }
        
        self.ctx.max_electrons = int(parameter_dict.get('max_electrons')) if 'max_electrons' in parameter_dict else None
        self.ctx.max_volume = float(parameter_dict.get('max_volume')) if 'max_volume' in parameter_dict else None
        
        self.ctx.min_electrons = int(parameter_dict.get('min_electrons')) if 'min_electrons' in parameter_dict else None
        self.ctx.min_volume = float(parameter_dict.get('min_volume')) if 'min_volume' in parameter_dict else None
        
        self.ctx.structure = self.inputs.structure.get_pymatgen()

        assert self.ctx.a and self.ctx.b and self.ctx.c \
               or (self.ctx.max_electrons and len(self.ctx.val_electrons)) \
               or self.ctx.max_volume \
               or (self.ctx.min_electrons and len(self.ctx.val_electrons)) \
               or self.ctx.min_volume, \
            'You must either provide values for `a`, `b` and `c` ' \
            'or for `max_electrons` or `min_electrons`' \
            'or for `max_volume` or `min_volume` in the parameters dictionary.'

    def abc_not_defined(self):
        return self.ctx.a is None and self.ctx.b is None and self.ctx.c is None

    def determine_factors(self):
        volume = self.ctx.structure.volume

        max_volume = None
        min_volume = None
        if self.ctx.max_electrons:
            electrons = sum(n * self.ctx.val_electrons.get(symbol)
                                     for symbol, n in self.ctx.structure.composition.as_dict().items())
            max_volume = volume * self.ctx.max_electrons / electrons
        if self.ctx.min_electrons:
            electrons = sum(n * self.ctx.val_electrons.get(symbol)
                                     for symbol, n in self.ctx.structure.composition.as_dict().items())
            min_volume = volume * self.ctx.min_electrons / electrons
        if self.ctx.max_volume:
            max_volume = self.ctx.max_volume
        if self.ctx.min_volume:
            min_volume = self.ctx.min_volume

        if max_volume:
            replicas = self.__calculate_factors(target_volume=max_volume, 
                                                                          cell_matrix=self.ctx.structure.lattice.matrix, 
                                                                          as_max=True)
        if min_volume:
            replicas = self.__calculate_factors(target_volume=min_volume, 
                                                                          cell_matrix=self.ctx.structure.lattice.matrix, 
                                                                          as_max=False)
        
        replicas[np.where(replicas == 0)] = 1
        self.ctx.a, self.ctx.b, self.ctx.c = replicas

    def replicate(self):
        self.out('structure',
                 StructureData(pymatgen=self.ctx.structure * np.array([self.ctx.a, self.ctx.b, self.ctx.c], dtype=int)))
                
    def __calculate_factors(self, target_volume, cell_matrix, as_max):
        norm_cell_matrix = []
        cell_lengths = []
        for i in [0,1,2]:
            norm_cell_matrix.append(np.array(cell_matrix[i])/np.linalg.norm(cell_matrix[i]))
            cell_lengths.append(np.linalg.norm(cell_matrix[i]))
        volume_norm_cell = np.linalg.det(norm_cell_matrix)
        target_edge = np.power(target_volume / volume_norm_cell, 1.0/3)
        target_factors = target_edge / np.array(cell_lengths)
        if as_max:
            factors = np.ceil(target_factors).astype(int)
            sign = -1
        else:
            factors = np.floor(target_factors).astype(int)
            sign = 1
        
        new_cell_matrix = []
        cell_lengths_new = []
        for i in [0,1,2]:
            new_cell_matrix.append(np.array(cell_matrix[i])*factors[i])
            cell_lengths_new.append(np.linalg.norm(new_cell_matrix[i]))
        volume_new_cell = np.linalg.det(new_cell_matrix)
        n_round = 0
        while (sign*volume_new_cell) < (sign*target_volume):
            n_round += 1
            change_side = 0
            edge_difference = np.abs(cell_lengths_new[0] + sign*cell_lengths[0] - target_edge)
            for i in [1,2]:
                edge_difference_test = np.abs(cell_lengths_new[i] + sign*cell_lengths[i] - target_edge)
                if edge_difference_test < edge_difference:
                    change_side = i
                    edge_difference = edge_difference_test
            factors[change_side] += sign
            for i in [0,1,2]:
                new_cell_matrix[i] = np.array(cell_matrix[i])*factors[i]
                cell_lengths_new[i] = np.linalg.norm(new_cell_matrix[i])
            volume_new_cell = np.linalg.det(new_cell_matrix)
            if n_round >= 10:
                break
 
        return list(factors)
