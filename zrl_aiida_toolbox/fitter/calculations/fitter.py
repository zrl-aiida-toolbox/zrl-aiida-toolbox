import json, itertools, yaml, os

import numpy as np
from aiida.common.datastructures import CalcInfo, CodeInfo

from aiida.common.utils import classproperty
from aiida.orm import JobCalculation, DataFactory

from pymatgen import Element

ParameterData = DataFactory('parameter')
StructureData = DataFactory('structure')
ArrayData = DataFactory('array')
List = DataFactory('list')
Float = DataFactory('float')
PotentialData = DataFactory('zrl.fitter.potential')
ParameterData = DataFactory('parameter')

class FitterCalculation(JobCalculation):
    def _init_internal_params(self):
        super(FitterCalculation, self)._init_internal_params()

    @classproperty
    def _use_methods(cls):
        retdict = JobCalculation._use_methods
        retdict.update({
            "parameters": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'parameters',
                'docstring': ''
            },
            "force_field": {
                'valid_types': ParameterData,
                'additional_parameter': None,
                'linkname': 'force_field',
                'docstring': ''
            },
            "structures": {
                'valid_types': StructureData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_structures_linkname,
                'docstring': "",
            },
            # "potential": {
            #     'valid_types': PotentialData,
            #     'additional_parameter': None,
            #     'linkname': 'potential',
            #     'docstring': ""
            # },
            "forces": {
                'valid_types': ArrayData,
                'additional_parameter': 'uuid',
                'linkname': cls._get_forces_linkname,
                'docstring': "",
            },
            # "stress": {
            #     'valid_types': List,
            #     'additional_parameter': 'uuid',
            #     'linkname': cls._get_stress_linkname,
            #     'docstring': "",
            # },
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
        inputs = dict(parameters=self.__prepare_parameters(inputdict),
                      force_field=self.__prepare_force_field(inputdict),
                      output=dict(population='progress.npy', append=False, best='best.yml'),
                      structures=[])

        species = list(inputs.get('force_field').get('species').keys())
        copy_list = [(tempfolder.get_abs_path('aiida.yml'), '.')]
        
        tempfolder.get_subfolder('./data', create=True)
        
        formulas = {}
        for key in inputdict:
            if 'structure' in key:
                uuid = key.replace('structure_', '')
                pmg = inputdict.get(key).get_pymatgen()
                formula = pmg.formula.replace(' ', '')
                formulas.setdefault(formula, 0)
                filename = tempfolder.get_abs_path('./data/%s.%03d.npy' % (formula, formulas.get(formula)))
                formulas[formula] += 1
                
                with open(filename, 'wb') as file:
                    data = np.array([
                        (
                            species.index(
                                site.specie.value 
                                if isinstance(site.specie, Element)
                                else site.specie.element.value
                            ), 
                            0 if isinstance(site.specie, Element)
                            else site.specie.oxi_state, 
                            site.coords[0], 
                            site.coords[1], 
                            site.coords[2], 
                            forces[0], 
                            forces[1], 
                            forces[2]
                        )
                        for site, forces in zip(pmg, inputdict.get('forces_%s' % uuid).get_array('forces'))
                    ])
                    np.save(file, pmg.lattice.matrix)
                    np.save(file, data)
                    np.save(file, inputdict.get('energy_%s' % uuid).value)

                copy_list += [
                    (str(filename), './data')
                ]

        
        for formula, count in formulas.items():
            inputs.get('structures').append(dict(
                format='./data/%s.%%03d.npy' % formula,
                range=[0, count]
            ))
        
        with open(tempfolder.get_abs_path('aiida.yml'), 'w') as f:
            f.write(yaml.dump(inputs))

        print tempfolder.get_abs_path('aiida.yml'), os.path.exists(tempfolder.get_abs_path('aiida.yml'))
        
        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = copy_list
        calcinfo.retrieve_list = [
            'progress.npy', 'best.yml'
        ]
        
        codeinfo = CodeInfo()
        codeinfo.code_uuid = inputdict.get('code').uuid
        codeinfo.cmdline_params = ['aiida.yml', 'genetic']
        codeinfo.stdout_name = 'aiida.stdout'
        codeinfo.stderr_name = 'aiida.stderr'

        calcinfo.codes_info = [codeinfo]

        return calcinfo

    def __prepare_force_field(self, inputdict):
        parameters = inputdict.get('force_field').get_dict()
        
        force_field_dict = {}
        force_field_dict['unit_charge'] = self.__float_or_float_list(parameters.get('unit_charge', 1), 2)
        force_field_dict['species'] = {}
        
        for key in inputdict:
            if 'structure' in key:
                for species in inputdict.get(key).get_pymatgen().composition:
                    el = str(species.value if isinstance(species, Element) else species.element.value)
                    if el not in force_field_dict.get('species'):
                        force_field_dict.get('species')[el] = \
                            parameters.get('charges', {}).get(el, Element(el).data.get('Common oxidation states', [0])[0])
                
        for element, shell in parameters.get('shells').items():
            shell_dict = force_field_dict\
                .get('shells', force_field_dict.setdefault('shells', {}))\
                .setdefault(element, {})
            
            shell_dict['k'] = shell['k']
            shell_dict['q'] = shell['q']
            
        for pair in parameters.get('pairs', []):
            pair_list = force_field_dict.get('pairs', force_field_dict.setdefault('pairs', []))
            
            if set(pair.keys()) >= {'a', 'rho', 'c'}:
                params = ('a', 'rho', 'c')
            elif set(pair.keys()) >= {'epsilon', 'alpha', 'rm'}:
                params = ('epsilon', 'alpha', 'rm')
            else:
                raise Exception('Invalid pair input')
                
            pair_list.append(
                dict(
                    species=pair.get('species'), 
                    **{
                        param: self.__float_or_float_list(pair.get(param), 2)
                        for param in params
                    }
                )
            )
        
        return force_field_dict
        
    def __prepare_parameters(self, inputdict):
        parameters = inputdict.get('parameters').get_dict()
        
        return dict(algorithm=parameters.get('algorithm', 'sade'),
                    verbosity=parameters.get('verbosity', 10),
                    save_every=parameters.get('save_every', 501),
                    population=parameters.get('population', 100),
                    max_steps=parameters.get('max_steps', 50000),
                    core_ftol=parameters.get('core_ftol', 1.e-1),
                    weights={
                        function: parameters.get('weights', {}).get(function, 1)
                        for function in ['forces', 'energy', 'f_corr']
                    })
    
    def __float_or_float_list(self, value, max_length=-1):
        if not isinstance(value, (list, tuple)):
            return float(value)
        
        itr = range(max_length) if max_length >= 0 else itertools.count(0)
        return [
            float(v)
            for i, v in zip(itr, value)
        ]
    