import re

import enum
from aiida.common.hashing import make_hash

from aiida.orm.data import Data


class PotentialData(Data):
    @enum.unique
    class PairType(enum.Enum):
        BUCK = 'buck'
        BUCK_COUL_CUT = 'buck/coul/cut'
        BUCK_COUL_LONG = 'buck/coul/long'
        BUCK_COUL_LONG_CS = 'buck/coul/long/cs'
        BUCK_COUL_MSM = 'buck/coul/msm'
        BUCK_LONG_COUL_LONG = 'buck/long/coul/long'
        LJ_CUT = 'lj/cut'
        LJ_CUT_COUL_CUT = 'lj/cut/coul/cut'
        LJ_CUT_COUL_DEBYE = 'lj/cut/coul/debye'
        LJ_CUT_COUL_DSF = 'lj/cut/coul/dsf'
        LJ_CUT_COUL_LONG = 'lj/cut/coul/long'
        LJ_CUT_COUL_LONG_CS = 'lj/cut/coul/long/cs'
        LJ_CUT_COUL_MSM = 'lj/cut/coul/msm'
        LJ_CUT_DIPOLE_CUT = 'lj/cut/dipole/cut'
        LJ_LONG_COUL_LONG = 'lj/long/coul/long'
        
    @enum.unique
    class BondType(enum.Enum):
        HARMONIC = 'harmonic'

    def __init__(self, pair_type=None, bond_type=None, unit_charge=1, charges=dict(), pairs=[], shells={}, bonds=[], dbnode=None):
        super(PotentialData, self).__init__(dbnode=dbnode)
        if dbnode:
            return
        self.pair_type = pair_type
        self.bond_type = bond_type
        self.unit_charge = unit_charge
        self.charges = charges
        for pair in pairs:
            self.set_pair(pair.get('kinds')[0], pair.get('kinds')[1 if len(pair.get('kinds')) > 1 else 0],
                          **dict([(key, value) for key, value in pair.items() if key != 'kinds']))
        for bond in bonds:
            self.set_bond(pair.get('kinds')[0], pair.get('kinds')[1 if len(pair.get('kinds')) > 1 else 0],
                          **dict([(key, value) for key, value in pair.items() if key != 'kinds']))
        for kind, args in shells.items():
            self.set_shell(kind, **args)

    @property
    def pair_type(self):
        pair_type = self.get_attr('pair_type', None)
        return self.__class__.PairType(pair_type) if pair_type else None

    @pair_type.setter
    def pair_type(self, pair_type):
        pair_type = self.__class__.PairType(pair_type)
        self._set_attr('pair_type', pair_type.value)
        
    @property
    def bond_type(self):
        bond_type = self.get_attr('bond_type', None)
        return self.__class__.BondType(bond_type) if bond_type else None

    @bond_type.setter
    def bond_type(self, bond_type):
        bond_type = self.__class__.BondType(bond_type)
        self._set_attr('bond_type', bond_type.value)

    @property
    def unit_charge(self):
        return self.get_attr('unit_charge')

    @unit_charge.setter
    def unit_charge(self, value):
        return self._set_attr('unit_charge', float(value))

    @property
    def charges(self):
        return self.get_attr('charges')

    @charges.setter
    def charges(self, value):
        self._set_attr('charges', {str(kind): float(charge) for kind, charge in value.items()})

    @property
    def pairs(self):
        return [
            dict(kinds=list(re.match('pair_([^_]+)_([^_]+)', key).groups()), **value)
            for key, value in dict(self.iterattrs()).items() if 'pair_' in key and key != 'pair_type'
        ]
    
    @property
    def bonds(self):
        return [
            dict(kinds=list(re.match('bond_([^_]+)_([^_]+)', key).groups()), **value)
            for key, value in dict(self.iterattrs()).items() if 'bond_' in key and key != 'bond_type'
        ]
    
    @property
    def shells(self):
        return {
            list(re.match('shell_([^_]+)', key).groups())[0]: dict(**value)
            for key, value in dict(self.iterattrs()).items() if 'shell_' in key
        }

    def set_pair(self, kind_1, kind_2, **kwargs):
        pair_type = self.pair_type
        assert pair_type is not None, "Set a pair type before adding a pair."
        if pair_type.value[:4] == 'buck':
            assert set(kwargs.keys()).issubset({'a', 'rho', 'c'})
            self._set_attr("pair_{}_{}".format(kind_1, kind_2), kwargs)
        if pair_type.value[:2] == 'lj':
            assert set(kwargs.keys()).issubset({'a', 'b'})
            self._set_attr("pair_{}_{}".format(kind_1, kind_2), kwargs)

    def delete_pair(self, kind_1, kind_2):
        self._del_attr("pair_{}_{}".format(kind_1, kind_2))

    def set_bond(self, kind_1, kind_2, **kwargs):
        bond_type = self.bond_type
        assert bond_type is not None, "Set a bond type before adding a pair."
        assert set(kwargs.keys()).issubset({'k', 'r0'})
        self._set_attr("bond_{}_{}".format(kind_1, kind_2), kwargs)
        
    def delete_bond(self, kind_1, kind_2):
        self._del_attr("bond_{}_{}".format(kind_1, kind_2))

    def set_shell(self, kind, **kwargs):
        bond_type = self.bond_type
        assert bond_type is not None, "Set a bond type before adding a pair."
        assert set(kwargs.keys()).issubset({'k', 'm', 'q'})
        kwargs['r0'] = 0
        self._set_attr("shell_{}".format(kind), kwargs)
        
    def delete_shell(self, kind):
        self._del_attr("shell_{}".format(kind))

    def get_hash(self, ignore_errors=True, **kwargs):
        return make_hash(dict(self.iterattrs()))
