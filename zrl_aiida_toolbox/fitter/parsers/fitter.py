import json
import numpy as np

from aiida.parsers.parser import Parser
from aiida.orm import DataFactory, CalculationFactory

FitterCalculation = CalculationFactory('zrl.fitter')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
PotentialData = DataFactory('zrl.fitter.potential')

class FitterParser(Parser):
    def __init__(self, calc):
        if not isinstance(calc, FitterCalculation):
            raise Exception('Input calc must be a FitterCalculation.')

        super(FitterParser, self).__init__(calc)

    def parse_with_retrieved(self, retrieved):
        with open('%s/potential.restart' % retrieved.get('retrieved_temporary_folder')) as f:
            restart = json.load(f)

            force_field = restart.get('force_field')
            potential = PotentialData(pair_type=force_field.get('pair_type'),
                                      bond_type=force_field.get('bond_type'),
                                      unit_charge=force_field.get('unit_charge'),
                                      charges=force_field.get('charges'),
                                      pairs=force_field.get('pairs'),
                                      bonds=force_field.get('bonds'),
                                      shells=force_field.get('shells'))

        with open('%s/aiida.out' % retrieved.get('retrieved_temporary_folder')) as f:
            data = []
            for line in f:
                line = line.strip()
                if line and line[0] not in '[#':
                    data.append(np.fromstring(data, sep=' '))

            array = ArrayData()

            array.set_array('step', data[:, 0])
            array.set_array('energy', data[:, 1])
            array.set_array('stress', data[:, 2])
            array.set_array('forces', data[:, 3])
            array.set_array('forces_correlation', data[:, 4])
            array.set_array('costs', data[:, 5])

        return True, [('force_field', potential), ('cost', array)]
