import numpy as np
def front_tier_models ():
    def front_vertical_load(m, l_f, l_r, h, a, g):
        """
        Calculate the front vertical load F_zf.

        Parameters:
            m   (float): total vehicle mass [kg]
            l_f (float): distance from CoG to front axle [m]
            l_r (float): distance from CoG to rear axle [m]
            h   (float): center of gravity height [m]
            a   (float): longitudinal acceleration [m/s^2]
            g   (float): gravitational acceleration [m/s^2] (default: 9.81)

        Returns:
            float: front vertical load [F_zf]
        """
        return (m / (l_f + l_r)) * (g * l_r - h * a)
    
# 1. Vehicle parameters
m   = 1558.4            # mass [kg]
l_f = 2.611 * 0.41      # distance CoG->front axle [m]
l_r = 2.611 * 0.59      # distance CoG->rear axle [m]
h   = 0.65              # CoG height [m]
g   = 9.81              # gravity [m/s^2]

