import numpy as np

from aiida.orm import DataFactory, CalculationFactory, Code
from aiida.work.workchain import WorkChain, while_, ToContext

PotentialData = DataFactory('zrl.fitter.potential')
StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
Int = DataFactory('int')

FitterCalculation = CalculationFactory('zrl.fitter')


class FitterWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(WorkChain, cls).define(spec)
        spec.input('parameters', valid_type=ParameterData)

        spec.input_namespace('structures', valid_type=StructureData, dynamic=True)
        spec.input_namespace('structures.forces', valid_type=ArrayData, dynamic=True)
        spec.input_namespace('structures.energy', valid_type=ParameterData, dynamic=True)
        spec.input_namespace('structures.stress', valid_type=ParameterData, dynamic=True)

        spec.input_namespace('forces', dynamic=True)
        spec.input('forces.code', valid_type=Code)
        spec.input('forces.settings', valid_type=ParameterData)

        spec.input_namespace('fitter')
        spec.input('fitter.code', valid_type=Code)
        spec.input('fitter.parameters', valid_type=ParameterData)
        spec.input('fitter.bounds', valid_type=ParameterData)
        spec.input('fitter.weights', valid_type=ParameterData)
        spec.input('fitter.options', valid_type=ParameterData)

        # spec.input('force_field', valid_type=PotentialData)

        spec.outline(
            cls.validate_inputs
        )

        # ,
        # while_(cls.converging)(
        #     cls.fit,
        #     cls.process
        # ),
        # cls.finalize

        # spec.output('force_field', valid_type=PotentialData)

    def validate_inputs(self):
        parameter_dict = self.inputs.parameters.get_dict()

        # self.ctx.step = 0
        # self.ctx.max_steps = parameter_dict.get('max_steps', 10)

        # self.ctx.force_field = self.inputs.force_field

        # self.ctx.delta = np.inf

        self.ctx.structures = {}
        self.ctx.energy = {}
        self.ctx.stress = {}
        self.ctx.forces = {}
        for uuid, structure in self.inputs.structures.items():
            self.ctx.structures[uuid] = structure
            self.ctx.energy[uuid] = self.inputs.energy[uuid]
            self.ctx.stress[uuid] = self.inputs.stress[uuid]
            self.ctx.forces[uuid] = self.inputs.forces[uuid]

    def converging(self):
        return np.abs(self.ctx.delta) > 1e-8 and self.ctx.step < self.ctx.max_steps

    def fit(self):
        future = self.submit(FitterCalculation,
                             code=self.inputs.fitter.code,
                             structures=self.ctx.structures,
                             force_field=self.ctx.force_field,
                             forces=self.ctx.forces,
                             energy=self.ctx.energy,
                             stress=self.ctx.stress,
                             bounds=self.inputs.fitter.bounds,
                             parameters=self.inputs.fitter.parameters,
                             weights=self.inputs.fitter.weights,
                             options=self.inputs.fitter.options.get_dict())

        return ToContext(run=future)
        pass

    def process(self):
        self.ctx.force_field = self.ctx.run.get_outputs(node_type=PotentialData)[0]
        data = self.ctx.run.get_outputs(node_type=ArrayData)[0]

        self.ctx.delta = data.get_array('costs')[-1] - data.get_array('costs')[-2]

        self.report(self.ctx.delta)

        self.ctx.step += 1

    def finalize(self):
        self.out('force_field', self.ctx.force_field)
