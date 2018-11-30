# zrl-aiida-toolbox
ZRL provided toolbox for [AiiDA](http://www.aiida.net/) developped within 
the [NCCR Marvel](http://nccr-marvel.ch/).

## Requirements

The 1.0 alpha version of AiiDA is currently required.

## Installation

```bash
pip install git+https://github.com/zrl-aiida-toolbox/zrl-aiida-toolbox.git@dev#egg=zrl-aiida-toolbox
```

## Table of content

- [Data types](#data-types)
- [Workchains](#workchains)
- [Calculations](#calculations)

## <a name="data-types"></a>Data types

### PotentialData (*zrl.fitter.potential*)

#### `pair_type` *get/set* `str` 
Stores the type of short range pair potential.

#### `bond_type` *get/set* `str` 
Stores the type of the bond potential.

#### `unit_charge` *get/set* `float` 
Stores the unit charge.

#### `charges` *get/set* `{str: float}`
Stores for each element, the valence charges (atom charge = unit charge * valence charge).

#### `pairs` *get*
Tuple of dictionary containing the parameters of the pairs. The content of the dictionary 
will change based on the pair potential type, `pair_type`.

#### `bonds` *get*
Tuple of dictionary containing the parameters of the bonds. The content of the dictionary 
will change based on the bond potential type, `bond_type`.

#### `set_pair(kind_1, kind_2, **kwargs)`
Add a pair to the force field between elements `kind_1` and `kind_2` with parameters `**kwargs`.
The parameters will vary based on your `pair_type`.

#### `delete_pair(kind_1, kind_2)`
Delete the pair between `kind_1` and `kind_2`.

#### `set_bond(kind_1, kind_2, **kwargs)`
Add a bond to the force field between elements `kind_1` and `kind_2` with parameters `**kwargs`.
The parameters will vary based on your `bond_type`.

#### `delete_bond(kind_1, kind_2)`
Delete the bond between `kind_1` and `kind_2`.


## <a name="workchains"></a>Workchains

- [zrl.utils.replicate](#zrl-utils-replicate)
- [zrl.utils.partial_occ](#zrl-utils-partial_occ)
- [zrl.utils.shake](#zrl-utils-shake)

- [zrl.fitter](#zrl-fitter-workchain)

### <a name="zrl-utils-replicate"></a>ReplicateWorkChain (*zrl.utils.replicate*)

This workchain creates a supercell based on the provided criteria. Currently supported 
criteria are, in order of precedence, explicit supercell size, maximum number of electrons 
and maximum volume.

**Inputs:**
- structure: `StructureData`
- parameters: `ParameterData`

**Outputs:**
- structure: `StructureData`

The `parameters` input expects the following parameters:
- a, b and c `int`: explicit supercell dimensions.
- val_electrons `{str: int}`: dictionary containting the number of valence electrons used 
  for each species. 
- max_electrons `int`: maximum number of valence electrons requested
- max_volume `float`: maximum volume requested.

### <a name="zrl-utils-partial_occ"></a>PartialOccupancyWorkChain (*zrl.utils.partial_occ*)

This workchain converts a structure containing partial occupancies into realisations of these
partial occupancies respecting the fractional occupancies of the sites. The returned 
realisations are selected to minimize the Coulomb interactions between the partial sites 
assuming point-charge particles using a Monte-Carlo approach.

**Inputs:**
- structure: `StructureData`
- parameters: `ParameterData`
- seed: `Int` (optional)

**Outputs:**
- structures: `[StructureData]`
- seed: `Int`
- energy: `ArrayData`

The `parameters` input expects the following parameters:
- charges `{str: float}`: a dictionary containing the charges to be used for the energy 
  calculation
- vacancy_ion `str`: the symbol of the element to be used as vacancy. By default, the workchain 
  uses `Lr`. This does not need to be changed unless there is a conflict.
- n_conf_target `int`: the number of realizations requested. The workchain will always return at
  most the requested number of generated structures (default: 1).
- equilibration `int`: the number of steps used to equilibrate before the sampling (default: 10).
- pick_conf_every `int`: the number of steps between two structures being selected (default: 100).
- n_rounds `int`: the maximum number of rounds to execute, each round corresponding to the pick of
  one structure (default: `equilibration` + `pick_conf_every` + 10).
- temperature `float`: effective temperature for the Monte-Carlo selection (default: 1000).
- return_unique `bool`: flag controlling whether a structure can be returned multiple times 
  (default: True).
- selection `str`: selection method of the output structures. Options are *reservoir sampling* and *last* 
  (default: *reservoir sampling*).

### <a name="zrl-utils-shake"></a>ShakeWorkChain (*zrl.utils.shake*)

This workchain adds a normal noise on the atomic positions and/or the lattice vectors.

**Inputs:**
- structure: `StructureData`
- parameters: `ParameterData`
- seed: `Int` (optional)

**Outputs:**
- structures: `[StructureData]`
- seed: `Int`

The `parameters` input expects the following parameters:
- stdev_atms `float`: standard deviation of the normal distribution used for the atomic 
  positions. 
- stdev_cell `float`: standard deviation of the normal distribution used for the lattice 
  vectors. 
  positions. 
- n `int`:  number of structures to generate.

### <a name="zrl-fitter-workchain"></a>FitterWorkChain (*zrl.fitter*)

## Calculations

- [FitterCalculation](#zrl-fitter-calculation)

### <a name="zrl-fitter-calculation"></a>FitterCalculation (*zrl.fitter*)

**Inputs**
- structures: `{uuid: StructureData}`
- force_field: `PotentialData`
- bounds: `{str: (float, float)}`
- forces: `{uuid: ArrayData}`
- stress: `{uuid: ParameterData}`
- energy: `{uuid: energy}`
- parameters: `ParameterData`
- weights: `ParameterData`

**Outputs**
- force_field: `PotentialData`
- cost: `ArrayData`