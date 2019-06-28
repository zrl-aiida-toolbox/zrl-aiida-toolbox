import re

import enum
from aiida.common.hashing import make_hash

from aiida.orm.data import Data


class PotentialData(Data):
    def __init__(self, unit_charge=1, charges=None, pairs=None, shells=None, dbnode=None):
        super(PotentialData, self).__init__(dbnode=dbnode)
        if dbnode:
            return
        
        self.unit_charge = unit_charge
        self.charges = charges if isinstance(charges, dict) else {}
        if isinstance(pairs, (list, tuple)):
            for pair in pairs:
                self.set_pair(pair.get('species')[0], pair.get('species')[1 if len(pair.get('species')) > 1 else 0],
                              **dict([(key, value) for key, value in pair.items() if key != 'species']))
        if isinstance(shells, dict):
            for kind, args in shells.items():
                self.set_shell(kind, **args)
                
    @property
    def pair_variant(self):
        pair_variant = self.get_attr('pair_variant', None)
        return self.__class__.PairVariant(pair_variant) if pair_variant else None

    @pair_variant.setter
    def pair_variant(self, pair_variant):
        pair_variant = self.__class__.PairVariant(pair_variant)
        self._set_attr('pair_variant', pair_variant.value)
        
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
        self._set_attr('charges', { str(kind): float(charge) for kind, charge in value.items() })

    @property
    def pairs(self):
        return [
            dict(kinds=list(re.match('pair_([^_]+)_([^_]+)', key).groups()), **value)
            for key, value in dict(self.iterattrs()).items() if 'pair_' in key and key != 'pair_type'
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
        params = set(kwargs.keys())
    
        assert params <= {'a', 'rho', 'c'}
        
        self._set_attr("pair_{}_{}".format(kind_1, kind_2), { param: kwargs.get(param, 0) for param in ['a', 'rho', 'c'] })
        
    def delete_pair(self, kind_1, kind_2):
        self._del_attr("pair_{}_{}".format(kind_1, kind_2))

    def set_shell(self, kind, **kwargs):
        assert set(kwargs.keys()) >= {'k', 'm', 'q'}
        kwargs['r0'] = 0
        self._set_attr("shell_{}".format(kind), { param: kwargs.get(param) for param in ['k', 'm', 'q', 'r0'] })
        
    def delete_shell(self, kind):
        self._del_attr("shell_{}".format(kind))

    def get_hash(self, ignore_errors=True, **kwargs):
        return make_hash(dict(self.iterattrs()))
