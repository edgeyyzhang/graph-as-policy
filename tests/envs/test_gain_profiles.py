"""Controller lifecycle tests use real environment methods with lightweight controllers."""
from types import SimpleNamespace
import numpy as np
import pytest
from gap.runtime.motion_profile import ControllerGainOffsets
FrankaLiberoEnv = pytest.importorskip('gap.envs.libero_env').FrankaLiberoEnv


def env():
    e=FrankaLiberoEnv.__new__(FrankaLiberoEnv)
    e._osc_ctrl=SimpleNamespace(kp=np.full(6,150.),kd=np.full(6,2*np.sqrt(150.)),impedance_mode='fixed')
    e._joint_ctrl=SimpleNamespace(kp=np.full(7,50.),kd=np.full(7,2*np.sqrt(50.)),impedance_mode='fixed')
    e._joint_motion_mode='closed_loop'; e._gain_scope_depth=0
    e._capture_gain_baselines()
    return e


def test_independent_gains_nested_and_failure_restore():
    e=env(); original=[(c.kp.copy(),c.kd.copy()) for c in (e._osc_ctrl,e._joint_ctrl)]
    with pytest.raises(RuntimeError):
        with e.controller_gain_scope(ControllerGainOffsets(10.,-2.)):
            for c, (kp,kd) in zip((e._osc_ctrl,e._joint_ctrl),original):
                np.testing.assert_array_equal(c.kp,kp+10.)
                np.testing.assert_array_equal(c.kd,kd-2.)
            with e.controller_gain_scope(ControllerGainOffsets(0.,3.)):
                np.testing.assert_array_equal(e._joint_ctrl.kp,original[1][0])
                np.testing.assert_array_equal(e._joint_ctrl.kd,original[1][1]+3.)
            np.testing.assert_array_equal(e._joint_ctrl.kp,original[1][0]+10.)
            raise RuntimeError('failure')
    assert e._gain_scope_depth == 0
    for c,(kp,kd) in zip((e._osc_ctrl,e._joint_ctrl),original):
        np.testing.assert_array_equal(c.kp,kp); np.testing.assert_array_equal(c.kd,kd)


@pytest.mark.parametrize('offset', [ControllerGainOffsets(-51.,0.), ControllerGainOffsets(0.,-15.)])
def test_invalid_gain_rejected_before_mutation(offset):
    e=env()
    with pytest.raises(ValueError,match='positive'):
        with e.controller_gain_scope(offset): pass
    np.testing.assert_array_equal(e._osc_ctrl.kp,np.full(6,150.))
    assert e._gain_scope_depth == 0


def test_reset_and_nonblocking_rejected_in_scope():
    from gap.connector.core import Connector
    e=env(); c=Connector.__new__(Connector); c.env=e
    with e.controller_gain_scope(ControllerGainOffsets()):
        with pytest.raises(ValueError,match='reset'): e.reset()
        with pytest.raises(ValueError,match='Nonblocking'): c.move_to_joints([],max_steps=0)


def test_teleport_and_variable_impedance_rejected():
    e=env(); e._joint_motion_mode='teleport'
    with pytest.raises(ValueError,match='closed_loop'):
        with e.controller_gain_scope(ControllerGainOffsets()): pass
    e._joint_motion_mode='closed_loop'; e._osc_ctrl.impedance_mode='variable'
    with pytest.raises(ValueError,match='fixed'):
        with e.controller_gain_scope(ControllerGainOffsets()): pass
