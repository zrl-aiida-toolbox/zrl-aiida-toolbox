# zrl-aiida-toolbox
ZRL provided toolbox for [AiiDA](http://www.aiida.net/) developped within 
the [NCCR Marvel](http://nccr-marvel.ch/).

## Requirements

The 1.0 alpha version of AiiDA is currently required.

## Installation

```bash
pip install git+https://github.com/zrl-aiida-toolbox/zrl-aiida-toolbox.git#egg=zrl-aiida-toolbox
```

## Workchains

### Utils

#### zrl.utils.replicate

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

#### zrl.utils.partial_occ

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

The `parameters` input expects the following parameters:
- charges `{str: float}`: a dictionary containing the charges to be used for the energy 
  calculation
- vacancy `str`: the symbol of the element to be used as vacancy. By default, the workchain 
  uses `Lr`. This does not need to be changed unless there is a conflict.
- max_rounds `int`: the maximum number of Monte-Carlo steps requested.
- max_configurations `int`: the maximum number of realizations to return. The workchain will
  always return at most the `max_configurations`-last generated structures.

#### zrl.utils.shake

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
