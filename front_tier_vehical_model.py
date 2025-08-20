import pandas as pd
import numpy as np

class FrontTireVehicleModel:
    def __init__(self, m, l_r, l_f, h, a, g):
        self.m = m
        self.l_r = l_r
        self.l_f = l_f
        self.h = h
        self.a = a
        self.g = g
        self.f_zf = self.compute_vehicle_load()

    def compute_vehicle_load(self):
        return self.m * (self.l_r + self.l_f) * (self.g * self.l_r - self.h * self.a)

    def angular_velocity(self, f_mt, f_bt, f_xf, i_hs, n_hs, i_tlc, n_tlc, r_wh, i_df, J_whf, J_fd, J_gb, J_m):
        e_ft = f_mt * i_hs * n_hs
        net_torque = self.front_wheel_model(e_ft, f_bt, f_xf, i_tlc, n_tlc, r_wh, i_df)
        net_inertia = self.wheel_momentum_of_inertia(J_whf, J_fd, J_gb, i_tlc, J_m, i_hs)

        def integrate_angular_velocity(omega_prev, dt):
            angular_acceleration = net_torque / net_inertia
            omega_new = omega_prev + angular_acceleration * dt
            return omega_new

        return integrate_angular_velocity

    def front_wheel_model(self, e_ft, f_bt, f_xf, i_tlc, n_tlc, r_wh, i_df):
        return (e_ft * i_tlc * n_tlc) - f_bt - (self.f_zf * r_wh * i_df) + f_xf

    def wheel_momentum_of_inertia(self, J_whf, J_fd, J_gb, i_tlc, J_m, i_hs):
        return J_whf + J_fd + (J_gb * (i_tlc ** 2)) + (J_m * (i_tlc * i_hs) ** 2)

    def front_omega_motor(self, wf, i_tlc, i_hs):
        def angular_motor(angular_motor_prev, dt):
            angular_wf = wf(angular_motor_prev, dt)
            angular_motor_new = angular_wf * (i_tlc * i_hs)
            return angular_motor_new
        return angular_motor

    def front_traction_tire_slip(self, wf_value, v, r_wh):
        return self.slip_ratio(wf_value, v, r_wh)

    def front_traction_force(self, tire_slip):
        friction_coeff = self.friction_coefficient(tire_slip)
        return friction_coeff * self.f_zf

    def slip_ratio(self, wf, v, r_wh):
        TS = v - wf * r_wh
        MS_acceleration = wf * r_wh
        MS_braking = v
        if TS <= 0 and MS_acceleration >= abs(TS):
            return TS / MS_acceleration if MS_acceleration != 0 else 0
        elif TS > 0 and MS_braking >= abs(TS):
            return TS / MS_braking
        return 0

    def friction_coefficient(self, tire_slip):
        file_path = "tiremodel.xls"
        data = pd.read_excel(file_path)
        Slip = data.iloc[:, 0].values
        Roadmu = data.iloc[:, 1].values
        return np.interp(tire_slip, Slip, Roadmu)

    def get_outputs(self, wf_func, omega_prev, dt, v, r_wh, i_tlc, i_hs, tire_slip=None):
        # Compute motor angular velocity from wheel function
        w_fmotor_func = self.front_omega_motor(wf_func, i_tlc, i_hs)
        w_fmotor = w_fmotor_func(omega_prev, dt)

        # Tire slip
        if tire_slip is None:
            tire_slip = self.front_traction_tire_slip(w_fmotor / (i_tlc * i_hs), v, r_wh)

        # Traction force
        f_xf = self.front_traction_force(tire_slip)

        return {
            "front angular vel": w_fmotor,
            "front tire slip": tire_slip,
            "front tire traction force": f_xf,
            "front tire vehicle load": self.f_zf
        }
