import re

import enum
from aiida.common.hashing import make_hash

from aiida.orm.data import Data


class PotentialData(Data):
    def __init__(self, pair_type=None, bond_type=None, unit_charge=1, charges=dict(), pairs=[], bonds=[], dbnode=None):
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
            self.set_bond(bond.get('kinds')[0], bond.get('kinds')[1 if len(bond.get('kinds')) > 1 else 0],
                          **dict([(key, value) for key, value in bond.items() if key != 'kinds']))

    @property
    def pair_type(self):
        return self.get_attr('pairType', None)

    @pair_type.setter
    def pair_type(self, pair_type):
        self._set_attr('pairType', str(pair_type) if pair_type else None)

    @property
    def bond_type(self):
        return self.get_attr('bondType', None)

    @bond_type.setter
    def bond_type(self, bond_type):
        self._set_attr('bondType', str(bond_type) if bond_type else None)

    @property
    def unit_charge(self):
        return self.get_attr('unit_charge')

    @unit_charge.setter
    def unit_charge(self, value):
        self._set_attr('unit_charge', float(value))

    @property
    def charges(self):
        return self.get_attr('charges')

    @charges.setter
    def charges(self, value):
        self._set_attr('charges', {str(kind): float(charge) for kind, charge in value.items()})

    @property
    def pairs(self):
        return tuple(
            dict(kinds=list(re.match('pair_([^_]+)_([^_]+)', key).groups()), **value)
            for key, value in dict(self.iterattrs()).items() if 'pair_' in key
        )

    def set_pair(self, kind_1, kind_2, **kwargs):
        assert self.pair_type is not None, 'Set a pair type before adding a pair.'
        self._set_attr("pair_{}_{}".format(kind_1, kind_2), kwargs)

    def delete_pair(self, kind_1, kind_2):
        self._del_attr("pair_{}_{}".format(kind_1, kind_2))

    @property
    def bonds(self):
        return tuple(
            dict(kinds=list(re.match('bond_([^_]+)_([^_]+)', key).groups()), **value)
            for key, value in dict(self.iterattrs()).items() if 'bond_' in key
        )

    def set_bond(self, kind_1, kind_2, **kwargs):
        assert self.bond_type is not None, 'Set a bond type before adding a bond.'
        self._set_attr("bond_{}_{}".format(kind_1, kind_2), kwargs)

    def delete_bond(self, kind_1, kind_2):
        self._del_attr("bond_{}_{}".format(kind_1, kind_2))

    def get_hash(self, ignore_errors=True, **kwargs):
        return make_hash(dict(self.iterattrs()))
