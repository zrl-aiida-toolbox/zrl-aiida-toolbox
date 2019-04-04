import numpy as np

from copy import copy
from itertools import chain
from aiida.orm import DataFactory, CalculationFactory, Code, WorkflowFactory, LinkType
from aiida.work.workchain import WorkChain, while_, ToContext, if_

# PotentialData = DataFactory('zrl.fitter.potential')
StructureData = DataFactory('structure')
ParameterData = DataFactory('parameter')
ArrayData = DataFactory('array')
Int = DataFactory('int')
Bool = DataFactory('bool')
String = DataFactory('str')
KpointsData = DataFactory('array.kpoints')

# FitterCalculation = CalculationFactory('zrl.fitter')


class FitterWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(WorkChain, cls).define(spec)
        # spec.input('parameters', valid_type=ParameterData)
        
        spec.input('use_cache', valid_type=Bool, default=Bool(True))
        spec.input('seed', valid_type=Int, required=False)
        spec.input('verbose', valid_type=Bool, default=Bool(False))

        spec.input_namespace('structures', valid_type=StructureData, dynamic=True)
        spec.input_namespace('structures.forces', valid_type=ArrayData, dynamic=True, required=False)
        spec.input_namespace('structures.energy', valid_type=ParameterData, dynamic=True, required=False)
        spec.input_namespace('structures.stress', valid_type=ParameterData, dynamic=True, required=False)

        # spec.input_namespace('fitter')
        # spec.input('fitter.code', valid_type=Code)
        # spec.input('fitter.parameters', valid_type=ParameterData)
        # spec.input('fitter.bounds', valid_type=ParameterData)
        # spec.input('fitter.weights', valid_type=ParameterData)
        # spec.input('fitter.options', valid_type=ParameterData)
        
        spec.input_namespace('structure', required=False)
        spec.input_namespace('structure.replicate', dynamic=True, required=False)
        spec.input_namespace('structure.partial', dynamic=True, required=False)
        spec.input_namespace('structure.shake', dynamic=True, required=False)
        
        spec.input_namespace('force', dynamic=True, required=False)
        spec.input('force.code', valid_type=Code, required=False)
        spec.input('force.settings', valid_type=ParameterData, required=False)
        spec.input('force.parameters', valid_type=ParameterData, required=False)
        spec.input('force.pseudo_family', valid_type=String, required=False)
        spec.input('force.kpoints', valid_type=KpointsData, required=False)
        
        # spec.input('force_field', valid_type=PotentialData)

        spec.outline(
            cls.validate_inputs,
            if_(cls.do_replicate)(
                cls.execute_replicate,
                cls.handle_replicate
            ),
            if_(cls.do_partial)(
                cls.execute_partial,
                cls.handle_partial
            ),
            if_(cls.do_shake)(
                cls.execute_shake,
                cls.handle_shake
            )
        )

        # ,
        # while_(cls.converging)(
        #     cls.fit,
        #     cls.process
        # ),
        # cls.finalize

        spec.output('seed', valid_type=Int)

    def validate_inputs(self):
        self.ctx.energy = {}
        self.ctx.stress = {}
        self.ctx.forces = {}
        
        seed = self.inputs.get('seed', np.random.randint(2**16 - 1))
        self.ctx.rs = np.random.RandomState(seed=seed)
        self.out('seed', Int(seed))
        
        self.ctx.structures = {}
        self.ctx.partial_structures = {}
        for uuid, structure in self.inputs.structures.items():
            if not isinstance(structure, StructureData):
                continue
            
            if structure.get_pymatgen().num_sites == sum(structure.get_pymatgen().composition.as_dict().values()):
                self.ctx.structures[uuid] = structure
            else:
                self.ctx.partial_structures[uuid] = structure

        if 'energy' in self.inputs.structures:
            self.ctx.energy = {uuid: self.inputs.structures.energy[uuid]
                               for uuid in self.ctx.structures
                               if uuid in self.inputs.structures.energy 
                               and uuid in self.ctx.structures}
        
        if 'stress' in self.inputs.structures:
            self.ctx.stress = {uuid: self.inputs.structures.stress[uuid]
                               for uuid in self.ctx.structures
                               if uuid in self.inputs.structures.stress 
                               and uuid in self.ctx.structures}
        
        if 'forces' in self.inputs.structures:
            self.ctx.forces = {uuid: self.inputs.structures.forces[uuid]
                               for uuid in self.ctx.structures
                               if uuid in self.inputs.structures.forces 
                               and uuid in self.ctx.structures}    
    
    def do_replicate(self):
        return len(getattr(self.inputs.structure, 'replicate', {})) > 0
    
    def execute_replicate(self):
        futures = {}
        process = WorkflowFactory('zrl.utils.replicate')
        inputs = {
            key: value
            for key, value in self.inputs.structure.replicate.items()
            if key in process.get_description().get('spec').get('inputs').keys()
        }        
        inputs.setdefault('verbose', self.inputs.verbose)
        
        for uuid, structure in chain(self.ctx.structures.items(),
                                     self.ctx.partial_structures.items()):
            if uuid not in self.ctx.energy or uuid not in self.ctx.stress or uuid not in self.ctx.forces:
                futures[uuid] = self.submit(process, structure=structure, **inputs)
                
        return ToContext(**futures)
    
    def handle_replicate(self):
        uuids = list(
            chain(
                self.ctx.structures.keys(),
                self.ctx.partial_structures.keys()
            )
        )
        for uuid in uuids:
            if uuid in self.ctx:
                outputs = self.ctx.get(uuid).get_outputs(node_type=StructureData, link_type=LinkType.RETURN)
                structure = outputs[0]
                
                if uuid in self.ctx.structures:
                    del self.ctx.structures[uuid]
                    self.ctx.structures[structure.uuid] = structure
                else:
                    del self.ctx.partial_structures[uuid]
                    self.ctx.partial_structures[structure.uuid] = structure
                    
                del self.ctx[uuid]
                
        self.report(self.ctx.structures.keys())
                
    def do_partial(self):
        return len(getattr(self.inputs.structure, 'partial', {})) > 0
    
    def execute_partial(self):
        futures = {}
        process = WorkflowFactory('zrl.utils.partial_occ')
        
        inputs = {
            key: value
            for key, value in self.inputs.structure.partial.items()
            if key in process.get_description().get('spec').get('inputs').keys()
        }
        
        inputs.setdefault('verbose', self.inputs.verbose)
        uuids = list(self.ctx.partial_structures.keys())
        for uuid in uuids:
            if uuid in self.ctx.energy and uuid in self.ctx.stress and uuid in self.ctx.forces:
                continue
            inputs_ = copy(inputs)
            inputs_['structure'] = self.ctx.partial_structures.get(uuid)
            inputs_.setdefault('seed', Int(self.ctx.rs.randint(2**16 - 1))) 
            futures[uuid] = self.submit(process, **inputs_)
                
        return ToContext(**futures)
    
    def handle_partial(self):
        uuids = list(self.ctx.partial_structures.keys())
        for uuid in uuids:
            if uuid in self.ctx:
                structures = self.ctx.get(uuid).get_outputs(node_type=StructureData, link_type=LinkType.RETURN)
                del self.ctx.partial_structures[uuid]
                for structure in structures:
                    self.ctx.structures[structure.uuid] = structure
                del self.ctx[uuid]
        self.report(self.ctx.structures.keys())
            
    def do_shake(self):
        return len(getattr(self.inputs.structure, 'shake', {})) > 0
    
    def execute_shake(self):
        futures = {}
        process = WorkflowFactory('zrl.utils.shake')
        
        inputs = {
            key: value
            for key, value in self.inputs.structure.shake.items()
            if key in process.get_description().get('spec').get('inputs').keys()
        }
        
        inputs.setdefault('verbose', self.inputs.verbose)
        
        uuids = list(self.ctx.structures.keys())
        for uuid in uuids:
            if uuid in self.ctx.energy and uuid in self.ctx.stress and uuid in self.ctx.forces:
                continue
            inputs_ = copy(inputs)
            inputs_['structure'] = self.ctx.structures.get(uuid)
            inputs_.setdefault('seed', Int(self.ctx.rs.randint(2**16 - 1))) 
            futures[uuid] = self.submit(process, **inputs_)
                
        return ToContext(**futures)
    
    def handle_shake(self):
        uuids = list(self.ctx.structures.keys())
        for uuid in uuids:
            if uuid in self.ctx:
                structures = self.ctx.get(uuid).get_outputs(node_type=StructureData, link_type=LinkType.RETURN)
                for structure in structures:
                    self.ctx.structures[structure.uuid] = structure
                del self.ctx.structures[uuid]
                del self.ctx[uuid]
    
    def converging(self):
        return np.abs(self.ctx.delta) > 1e-8 and self.ctx.step < self.ctx.max_steps

    def fit(self):
        future = self.submit(FitterCalculation,
                             code=self.inputs.fitter.code,
                             structures=self.ctx.structures,
                             force_field=self.ctx.force_field,
                             forces=self.ctx.forces,
                             energy=self.ctx.energy,
                             stress=self.ctx.stress,
                             bounds=self.inputs.fitter.bounds,
                             parameters=self.inputs.fitter.parameters,
                             weights=self.inputs.fitter.weights,
                             options=self.inputs.fitter.options.get_dict())

        return ToContext(run=future)
        pass

    def process(self):
        self.ctx.force_field = self.ctx.run.get_outputs(node_type=PotentialData)[0]
        data = self.ctx.run.get_outputs(node_type=ArrayData)[0]

        self.ctx.delta = data.get_array('costs')[-1] - data.get_array('costs')[-2]

        self.report(self.ctx.delta)

        self.ctx.step += 1

    def finalize(self):
        self.out('force_field', self.ctx.force_field)
