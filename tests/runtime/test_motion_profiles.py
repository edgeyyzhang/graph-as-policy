"""Additive motion profiles exercise real dispatch and invocation lifetimes."""
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from gap_core.tools import ToolRegistry
from gap.runtime.executor import WorkflowExecutor
from gap.runtime.motion_profile import ControllerGainOffsets, MotionProfile, prepare_inputs
from gap_core.tools.schema import FieldInfo


def workflow(tmp_path, node, **extra):
    raw = dict(version=3, nodes={'move': node, 'done': {'type': 'end', 'status': 'success'}},
               edges=[['START', 'move'], ['move', 'done']], **extra)
    (tmp_path/'workflow.json').write_text(json.dumps(raw))
    return tmp_path


def test_direct_tool_default_and_scoped_gains(tmp_path):
    state = {'gain': 1.}
    calls = []
    def move(x: float = .2) -> dict:
        calls.append((x, state['gain']))
        return {'x': x}
    reg = ToolRegistry(); reg.register_callable('test.move', move, summary='move')
    @contextmanager
    def scope(gains):
        old = state['gain']; state['gain'] += gains.kp_offset
        try: yield
        finally: state['gain'] = old
    path = workflow(tmp_path, dict(type='tool', tool='test.move'))
    ex = WorkflowExecutor(path, tool_registry=reg, max_node_workers=1,
         motion_profiles={'move': MotionProfile(ControllerGainOffsets(2., 0.), {'x': .05})},
         controller_gain_scope=scope, trace_dir=tmp_path/'trace')
    try: ex.execute()
    finally: ex.close()
    assert calls == [(.25, 3.)]
    assert state['gain'] == 1.
    profile = json.loads((tmp_path/'trace/node_data/move/motion_profile.json').read_text())
    assert profile['nominal']['x'] == .2
    assert profile['effective']['x'] == .25
    assert (tmp_path/'trace/node_data/move/iters/000/motion_profile.json').exists()


def test_script_nested_copy_and_unknown_field(tmp_path):
    (tmp_path/'move.py').write_text('''from typing import TypedDict
class Output(TypedDict):
    z: float
def run(ctx, pose: dict) -> Output:
    return {"z": pose["position"]["z"]}
''')
    node = dict(type='script', script='move.py', inputs={'pose': {'position': {'z': .3}}})
    workflow(tmp_path, node)
    for key, succeeds in [('pose.position.z', True), ('typo', False)]:
        ex = WorkflowExecutor(tmp_path, tool_registry=ToolRegistry(), max_node_workers=1,
             motion_profiles={'move': MotionProfile(input_offsets={key: .1})}, trace_dir=tmp_path/key)
        try:
            if succeeds:
                ex.execute()
                out = json.loads((tmp_path/key/'node_data/move/output.json').read_text())
                assert out['z'] == pytest.approx(.4)
            else:
                with pytest.raises(Exception, match='unknown profiled input'):
                    ex.execute()
        finally: ex.close()
    assert node['inputs']['pose']['position']['z'] == .3


def test_failure_restores_scope(tmp_path):
    active = []
    def fail(x: float) -> dict:
        assert active == [True]
        raise RuntimeError('intentional failure')
    reg = ToolRegistry(); reg.register_callable('test.fail', fail, summary='fail')
    @contextmanager
    def scope(_):
        active.append(True)
        try: yield
        finally: active.pop()
    workflow(tmp_path, dict(type='tool', tool='test.fail', inputs={'x': .2}))
    ex = WorkflowExecutor(tmp_path, tool_registry=reg, max_node_workers=1,
         motion_profiles={'move': MotionProfile(ControllerGainOffsets())}, controller_gain_scope=scope)
    try:
        with pytest.raises(Exception, match='intentional failure'): ex.execute()
    finally: ex.close()
    assert active == []


def test_profile_constraints_and_immutability(tmp_path):
    original = {'x': .1}; p = MotionProfile(input_offsets=original); original['x'] = 7.
    assert p.input_offsets['x'] == .1
    with pytest.raises(TypeError): p.input_offsets['x'] = 2.
    with pytest.raises(Exception): ControllerGainOffsets(float('nan'))
    reg = ToolRegistry(); reg.register_callable('test.move', lambda: {}, summary='move')
    workflow(tmp_path, dict(type='tool', tool='test.move'))
    with pytest.raises(ValueError, match='max_node_workers'):
        WorkflowExecutor(tmp_path, tool_registry=reg, motion_profiles={'move': p})
    with pytest.raises(ValueError, match='Unknown'):
        WorkflowExecutor(tmp_path, tool_registry=reg, max_node_workers=1, motion_profiles={'typo': p})
    workflow(tmp_path, dict(type='tool', tool='test.move', streaming=True))
    with pytest.raises(ValueError, match='streaming'):
        WorkflowExecutor(tmp_path, tool_registry=reg, max_node_workers=1, motion_profiles={'move': p})


def test_repeat_adjustments_do_not_compound_and_integers_rejected():
    fields = {'x': FieldInfo('x', float, 'float', True)}
    nominal = {'x': .3}
    profile = MotionProfile(input_offsets={'x': .1})
    for _ in range(3):
        assert prepare_inputs(profile, nominal, fields, ctx=None)['x'] == pytest.approx(.4)
    assert nominal == {'x': .3}
    assert prepare_inputs(profile, {'x': 0}, fields, ctx=None)['x'] == .1
    integer_fields = {'x': FieldInfo('x', int, 'int', True)}
    with pytest.raises(Exception, match='continuous'):
        prepare_inputs(profile, {'x': 3}, integer_fields, ctx=None)


def test_official_transport_automatic_height_and_outputs():
    root = Path(__file__).resolve().parents[3]
    source = root/'open-robot-skills/skills/transporting-objects/scripts/waypoint_move.py'
    spec = importlib.util.spec_from_file_location('profile_transport_test', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    def tool(name, **kwargs):
        if name == 'robot.describe_workspace': return dict(transport_z=.35, surface_z=.0, align_clearance_m=.1)
        if name == 'robot.describe_arm': return dict(arm_id=0)
        if name == 'robot.get_observation': return dict(arms=[dict(ee_pose=dict(position=dict(x=.0,y=.0,z=.2)))])
        if name == 'robot.grasp_frame': return dict(rotation=dict(w=1.,x=0.,y=0.,z=0.))
        if name == 'robot.go_to_pose_cartesian': calls.append(kwargs['pose']); return {}
        raise AssertionError(name)
    from gap_core.tools.schema import extract_schema
    ctx=SimpleNamespace(tool=tool)
    profile=MotionProfile(input_offsets={'safe_height': .02, 'drop_x': .01})
    values=prepare_inputs(profile, dict(drop_x=.1,drop_y=.2), extract_schema(module).inputs,
          ctx=ctx, resolver=module.resolve_motion_profile_inputs, validator=module.validate_motion_profile_inputs)
    result=module.run(ctx, **values)
    assert calls[0]['position']['z'] == pytest.approx(.37)
    assert calls[1]['position']['x'] == pytest.approx(.11)
    assert result['commanded_drop_x'] == pytest.approx(.11)
    with pytest.raises(ValueError, match='positive explicit'):
        prepare_inputs(MotionProfile(input_offsets={'safe_height': -1.}), dict(drop_x=.1,drop_y=.2),
            extract_schema(module).inputs, ctx=ctx, resolver=module.resolve_motion_profile_inputs,
            validator=module.validate_motion_profile_inputs)


def test_public_execute_profiles(tmp_path, monkeypatch):
    import gap
    import gap.skills
    monkeypatch.setattr(gap.skills, 'resolve_registries', lambda *a, **kw: [])
    calls=[]
    def move(x: float=.2) -> dict: calls.append(x); return {}
    reg=ToolRegistry(); reg.register_callable('test.move', move, summary='move')
    workflow(tmp_path, dict(type='tool', tool='test.move'))
    result=gap.execute(tmp_path, SimpleNamespace(tool_registry=reg), max_node_workers=1,
        motion_profiles={'move': MotionProfile(input_offsets={'x': .1})})
    assert result.success, result.error
    assert calls == [pytest.approx(.3)]


def test_subgraph_profile_and_downstream_ref(tmp_path):
    calls=[]
    def source(x: float=.2) -> dict: return {'x':x}
    def sink(x: float) -> dict: calls.append(x); return {'x':x}
    reg=ToolRegistry()
    reg.register_callable('test.source',source,summary='source')
    reg.register_callable('test.sink',sink,summary='sink')
    raw={'version':3,'nodes':{'run':{'type':'subgraph','ref':'sg'},'done':{'type':'end','status':'success'}},
         'edges':[['START','run'],['run','done']], 'subgraphs':{'sg':{
          'skill':'test','inputs':{},'outputs':{},
          'nodes':{'source':{'type':'tool','tool':'test.source'},
                   'sink':{'type':'tool','tool':'test.sink','inputs':{'x':{'$ref':'source.x'}}},
                   'ok':{'type':'noop'}},
          'edges':[['START','source'],['source','sink'],['sink','ok'],['ok','END']],
          'exit':{'router_field':None,'success_values':['ok']}}}}
    (tmp_path/'workflow.json').write_text(json.dumps(raw))
    ex=WorkflowExecutor(tmp_path,tool_registry=reg,max_node_workers=1,
        motion_profiles={'sg.source':MotionProfile(input_offsets={'x':.05})})
    try: ex.execute()
    finally: ex.close()
    assert calls == [pytest.approx(.25)]


def test_dynamic_profiles_each_visit_and_trace(tmp_path):
    calls=[]
    def move(x: float=.2) -> dict:
        calls.append(x)
        return {'route':'again' if len(calls)<2 else 'done'}
    reg=ToolRegistry(); reg.register_callable('test.move',move,summary='move')
    raw={'version':3,'nodes':{'move':{'type':'tool','tool':'test.move'},'done':{'type':'end','status':'success'}},
         'edges':[['START','move']], 'conditional_edges':{'move':{'router_field':'route','mapping':{'again':'move','done':'done'}}}}
    (tmp_path/'workflow.json').write_text(json.dumps(raw))
    visits=[]
    def resolve(name,node,inputs):
        visits.append(name)
        return MotionProfile(input_offsets={'x': .1*len(visits)})
    ex=WorkflowExecutor(tmp_path,tool_registry=reg,max_node_workers=1,motion_profile_resolver=resolve,trace_dir=tmp_path/'trace')
    try: ex.execute()
    finally: ex.close()
    assert visits == ['move','move']
    assert calls == [pytest.approx(.3),pytest.approx(.4)]
    histories=list((tmp_path/'trace/node_data/move/iters').glob('*/motion_profile.json'))
    assert len(histories)==2


def test_public_execute_resolver_in_subgraph_and_bad_profile_is_not_routed(tmp_path, monkeypatch):
    import gap
    import gap.skills
    from gap_core.errors import is_terminal
    monkeypatch.setattr(gap.skills, 'resolve_registries', lambda *a, **kw: [])
    calls=[]
    def move(x: float=.2) -> dict: calls.append(x); return {}
    reg=ToolRegistry(); reg.register_callable('test.move', move, summary='move')
    raw={'version':3,'nodes':{'run':{'type':'subgraph','ref':'sg'},
                              'done':{'type':'end','status':'success'},'abort':{'type':'end','status':'failure'}},
         'edges':[['START','run']],
         'conditional_edges':{'run':{'router_field':'exit','mapping':{'ok':'done','failed':'abort'}}},
         'subgraphs':{'sg':{'skill':'test','inputs':{},'outputs':{},
          'nodes':{'move':{'type':'tool','tool':'test.move'},'ok':{'type':'noop'}},
          'edges':[['START','move'],['move','ok'],['ok','END']],
          'exit':{'router_field':None,'success_values':['ok']},'on_error':'failed'}}}
    (tmp_path/'workflow.json').write_text(json.dumps(raw))
    seen=[]
    def resolver(full_id, node, resolved):
        seen.append(full_id)
        return MotionProfile(input_offsets={'x': .1})
    result=gap.execute(tmp_path, SimpleNamespace(tool_registry=reg), max_node_workers=1,
        motion_profile_resolver=resolver, trace_dir=tmp_path/'good')
    assert result.success, result.error
    assert seen == ['sg.move'] and calls == [pytest.approx(.3)]
    # A profile that cannot be applied is a configuration error: the subgraph must
    # not take its on_error exit and end the workflow as an ordinary failure.
    result=gap.execute(tmp_path, SimpleNamespace(tool_registry=reg), max_node_workers=1,
        motion_profile_resolver=lambda *a: MotionProfile(input_offsets={'typo': .1}), trace_dir=tmp_path/'bad')
    assert not result.success and result.exit_status is None
    assert is_terminal(result.error)
    assert len(calls) == 1
