import json

import numpy as np
from aiida.common.datastructures import CalcInfo, CodeInfo
from ase.io import write

from aiida.common.utils import classproperty
from aiida.orm import JobCalculation, DataFactory

ParameterData = DataFactory('parameter')
StructureData = DataFactory('structure')
ArrayData = DataFactory('array')
PotentialData = DataFactory('zrl.fitter.potential')


class FitterCalculation(JobCalculation):
    def _init_internal_params(self):
        super(FitterCalculation, self)._init_internal_params()

        self._default_parser = 'zrl.fitter'

    @classproperty
    def _use_methods(cls):
        retdict = JobCalculation._use_methods
        retdict.update({
            'structures': {
                'valid_types': StructureData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_structures_linkname,
                'docstring': '',
            },
            'force_field': {
                'valid_types': PotentialData,
                'additional_parameter': None,
                'linkname': 'force_field',
                'docstring': ''
            },
            'bounds': {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'bounds',
                'docstring': ''
            },
            'forces': {
                'valid_types': ArrayData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_forces_linkname,
                'docstring': '',
            },
            'stress': {
                'valid_types': ParameterData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_stress_linkname,
                'docstring': '',
            },
            'energy': {
                'valid_types': ParameterData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_energy_linkname,
                'docstring': '',
            },
            'parameters': {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'parameters',
                'docstring': '',
            },
            'weights': {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'weights',
                'docstring': '',
            }
        })
        return retdict

    @classmethod
    def _get_structures_linkname(self, uuid):
        return 'structure_%s' % uuid

    @classmethod
    def _get_forces_linkname(self, uuid):
        return 'forces_%s' % uuid

    @classmethod
    def _get_stress_linkname(self, uuid):
        return 'stress_%s' % uuid

    @classmethod
    def _get_energy_linkname(self, uuid):
        return 'energy_%s' % uuid

    def _prepare_for_submission(self, tempfolder, inputdict):
        inputs = dict(references=[],
                      **self.__prepare_input_dict(inputdict.get('parameters'),
                                                  inputdict.get('force_field'),
                                                  inputdict.get('bounds'),
                                                  inputdict.get('weights')))

        keys = ['energy', 'forces', 'stress']
        copy_list = [(tempfolder.get_abs_path('aiida.in'), '.')]
        for key in inputdict:
            if 'structure' in key:
                uuid = key.replace('structure_', '')
                inputs.get('references').append(uuid)
                input = tempfolder.get_abs_path(uuid)
                write((input + '.json').encode('utf8'), inputdict.get(key).get_ase(), format='json')

                headers = np.array(filter(lambda x: ('%s_%s' % (x, uuid)) in inputdict, keys))
                with open(input + '.npy', 'wb') as file:
                    np.save(file, headers)
                    for header in headers:
                        if header == 'forces':
                            np.save(file, inputdict.get('forces_%s' % uuid).get_array('forces')[0])
                            continue
                        np.save(file, inputdict.get('%s_%s' % (header, uuid, )).get_attr(header))
                copy_list += [
                    (input + '.json', '.'),
                    (input + '.npy', '.')
                ]

        with open(tempfolder.get_abs_path('aiida.in'), 'w') as f:
            json.dump(inputs, f, indent=1)

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = copy_list
        calcinfo.retrieve_temporary_list = ['aiida.restart', 'aiida.out']

        codeinfo = CodeInfo()
        codeinfo.withmpi = False
        codeinfo.code_uuid = inputdict.get('code').uuid
        codeinfo.cmdline_params = ['-n', '20', '-i', 'aiida.in', '-o', 'aiida.out', 'run']

        codeinfo.stdout_name = 'aiida.stdout'
        codeinfo.stderr_name = 'aiida.stderr'

        calcinfo.codes_info = [codeinfo]

        return calcinfo

    def __prepare_input_dict(self, parameters, force_field, bounds, weights):
        parameters = parameters.get_dict()
        weights = weights.get_dict()

        return {
            'fitter': {
                'algorithm': 'gradient',
                'output': 'aiida.out',
                'restart': {
                    'file': 'aiida.restart',
                    'save_only': True,
                    'frequency': parameters.get('restart', {}).get('frequency', 10)
                },
                'max_steps': parameters.get('max_steps', 100),
                'step_size': parameters.get('step_size', 1e-3)
            },
            'weights': {
                'costs': weights.get('costs', {}),
                'atoms': weights.get('atoms', {})
            },
            'bounds': bounds.get_dict(),
            'force_field': {
                'pair_type': force_field.pair_type,
                'bond_type': force_field.bond_type,
                'unit_charge': force_field.unit_charge,
                'charges': force_field.charges,
                'pairs': force_field.pairs,
                'bonds': force_field.bonds
            }
        }
