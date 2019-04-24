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
        with open('%s/best.yaml' % retrieved.get('retrieved_temporary_folder')) as f:
            best_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
            best = PotentialData(
                unit_charge=best_dict.get('unit_charge'),
                charges=best_dict.get('species'),
                shells=best_dict.get('shells')
            )
            
            for pair in best_dict.get('pairs'):
                best.set_pair(pair.get('species')[0], pair.get('species')[1], a=pair.get('a'), rho=pair.get('rho'), c=pair.get('c'))

        return True, [('force_field', potential)]
