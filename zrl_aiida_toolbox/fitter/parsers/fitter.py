# import json
# 


from aiida.parsers.parser import Parser
from aiida.orm import DataFactory, CalculationFactory

FitterCalculation = CalculationFactory('zrl.fitter')
PotentialData = DataFactory('zrl.fitter.potential')

class FitterParser(Parser):
    def __init__(self, calc):
        if not isinstance(calc, FitterCalculation):
            raise Exception('Input calc must be a FitterCalculation.')

        super(FitterParser, self).__init__(calc)

    def parse_with_retrieved(self, retrieved):
        import yaml
        from pymatgen import Specie
        import os
        import numpy as np
        from itertools import count

        ParameterData = DataFactory('parameter')
        ArrayData = DataFactory('array')
        
        try:
            out_folder = retrieved[self._calc._get_linkname_retrieved()]
        except KeyError:
            self.logger.error("No retrieved folder found")
            return False, ()
        
        best_file = os.path.join(out_folder.get_abs_path('.'), 'best.yml')
        progress_file = os.path.join(out_folder.get_abs_path('.'), 'progress.npy')
        
        ff_dict = dict(self._calc.get_inputs(ParameterData, also_labels=True)).get('force_field').get_dict()
        
        with open(best_file) as f:
            best_dict = yaml.load(f.read(), Loader=yaml.Loader)
            
            best = ParameterData(dict=dict(
                unit_charge=best_dict.get('unit_charge'),
                shells=best_dict.get('shells'),
                pairs=best_dict.get('pairs'),
                species=[
                    str(Specie(element, charge))
                    for element, charge in best_dict.get('species').items()
                ]
            ))
            
        params = {}    
        unit_charge = ff_dict.get('unit_charge')
        
        with open(progress_file, 'rb') as f:
            costs_array = ArrayData()
            costs = None
            try:
                while True:
                    xs = np.load(f)
                    local_costs = np.load(f)
                    if costs is None:
                        costs = np.empty((0, local_costs.size), dtype=np.float)
                    costs = np.insert(costs, costs.shape[0], local_costs, axis=0)
                    
                    col = iter(count())
                    params.setdefault('q', np.empty((0, local_costs.size), dtype=np.float))
                    if isinstance(unit_charge, (float, int)):
                        params['q'] = np.insert(params['q'], params['q'].shape[0], unit_charge * np.ones(local_costs.size))
                    else:
                        params['q'] = np.insert(params['q'], params['q'].shape[0], xs[:, next(col)])
                    
                    for pair in ff_dict.get('pairs'):
                        key = '_'.join(pair.get('species'))
                        params.setdefault(key, np.empty((0, local_costs.size, 3), dtype=np.float))
                        
                        a = pair.get('a')
                        rho = pair.get('rho')
                        c = pair.get('c')
                        
                        a = a * np.ones(local_costs.size) if isinstance(a, (int, float)) else xs[:, next(col)]
                        rho = rho * np.ones(local_costs.size) if isinstance(rho, (int, float)) else xs[:, next(col)]
                        c = c * np.ones(local_costs.size) if isinstance(c, (int, float)) else xs[:, next(col)]
                        
                        params[key] = np.insert(params[key], params[key].shape[0], np.array([a, rho, c]).T, axis=0)
            except IOError as e:
                costs_array.set_array('costs', costs)
                for key, value in params.items():
                    costs_array.set_array(key, value)
                
            # best = PotentialData(
            #     unit_charge=best_dict.get('unit_charge'),
            #     charges=best_dict.get('species'),
            #     shells=best_dict.get('shells')
            # )
            
            # for pair in best_dict.get('pairs'):
            #     best.set_pair(pair.get('species')[0], pair.get('species')[1], a=pair.get('a'), rho=pair.get('rho'), c=pair.get('c'))

        return True, [
            ('force_field', best), 
            ('costs', costs_array)
        ]
    
