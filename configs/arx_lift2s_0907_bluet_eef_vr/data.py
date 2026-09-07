from tau0_vla.data import register_config
from tau0_vla.adapters.arx_lift2s.calibrated_config import make_config

@register_config
def arx_lift2s_0907_bluet_eef_vr_ft():
    return make_config(__file__, 'BlueT', 'eef-vr')
