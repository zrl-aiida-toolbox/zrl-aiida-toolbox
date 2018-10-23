import json

import numpy as np
from aiida.common.datastructures import CalcInfo, CodeInfo
from ase.io import write

from aiida.common.utils import classproperty
from aiida.orm import JobCalculation, DataFactory

ParameterData = DataFactory('parameter')
StructureData = DataFactory('structure')
ArrayData = DataFactory('array')
List = DataFactory('list')
Float = DataFactory('float')
PotentialData = DataFactory('zrl.fitter.potential')


class FitterCalculation(JobCalculation):
    def _init_internal_params(self):
        super(FitCalculation, self)._init_internal_params()

    @classproperty
    def _use_methods(cls):
        retdict = JobCalculation._use_methods
        retdict.update({
            "structures": {
                'valid_types': StructureData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_structures_linkname,
                'docstring': "",
            },
            "potential": {
                'valid_types': PotentialData,
                'additional_parameter': None,
                'linkname': 'potential',
                'docstring': ""
            },
            "forces": {
                'valid_types': ArrayData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_forces_linkname,
                'docstring': "",
            },
            "stress": {
                'valid_types': List,
                'additional_parameter': 'uuid',
                'linkname': cls._get_stress_linkname,
                'docstring': "",
            },
            "energy": {
                'valid_types': Float,
                'additional_parameter': 'uuid',
                'linkname': cls._get_energy_linkname,
                'docstring': "",
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
        inputs = dict(fitter=self.__prepare_fitter_input(inputdict),
                      force_field=self.__prepare_force_field(inputdict),
                      references=[],
                      bounds=dict(a=[100, None], c=[1e-9, None],
                                  rho=[0.005, 0.995], q=[0.5, 1.5]))

        keys = ['forces', 'stress', 'energy']
        copy_list = [(tempfolder.get_abs_path('aiida.in'), '.')]
        for key in inputdict:
            if 'structure' in key:
                uuid = key.replace('structure_', '')
                inputs['references'].append(uuid)
                input = tempfolder.get_abs_path(uuid)
                write((input + '.json').encode('utf8'), inputdict.get(key).get_ase(), format='json')

                with open(input + '.npy', 'wb') as file:
                    np.save(file, keys)
                    np.save(file, inputdict.get('forces_%s' % uuid).get_array('forces')[0])
                    np.save(file, np.array(inputdict.get('stress_%s' % (uuid,))))
                    np.save(file, inputdict.get('energy_%s' % (uuid,)).value)

                copy_list += [
                    (input + '.json', '.'),
                    (input + '.npy', '.')
                ]

        with open(tempfolder.get_abs_path('aiida.in'), 'w') as f:
            json.dump(inputs, f, indent=3)

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = copy_list

        codeinfo = CodeInfo()
        codeinfo.withmpi = False
        codeinfo.code_uuid = inputdict.get('code').uuid
        codeinfo.cmdline_params = ['-n', '20', '-i', 'aiida.in', 'run']
        codeinfo.stdout_name = 'aiida.stdout'
        codeinfo.stderr_name = 'aiida.stderr'

        calcinfo.codes_info = [codeinfo]

        return calcinfo

    def __prepare_fitter_input(self, inputdict):
        return dict(algorithm='scipy')

    def __prepare_force_field(self, inputdict):
        potential = inputdict.get('potential')
        return dict(pair_type=potential.pair_type.value,
                    bond_type=potential.bond_type.value,
                    unit_charge=potential.unit_charge,
                    charges=potential.charges,
                    pairs=potential.pairs,
                    bonds=potential.bonds,
                    shells=potential.shells)
