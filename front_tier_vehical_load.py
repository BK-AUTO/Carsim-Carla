import pandas as pd
import numpy as np
def front_tire_vehical_load (m,l_r,l_f,h,a,g):
    f_zf = m*(l_r + l_f)*(g*l_r-h*a)
    return f_zf
# Calculates the front wheel angular velocity integration function for a vehicle, considering torque, forces, gear ratios, wheel parameters, and inertia.
    ''' 
        f_bt: front_brake_torque
        f_xf: front_tier_traction_forces
        f_mt: front_motor_torque
        f_zf: front_tire_vehical_load
        i_hs, n_hs: gear ratios
        i_tlc, n_tlc, r_wh, i_df: wheel model parameters
        J_whf, J_fd, J_gb, J_m: inertia parameters
    '''
    e_ft = f_mt * i_hs * n_hs    # e_ft: effective_front_torque

    def front_wheel_model(e_ft, f_bt, f_zf, f_xf, i_tlc, n_tlc, r_wh, i_df):
        net_torque = (e_ft * i_tlc * n_tlc) - f_bt - (f_zf * r_wh * i_df) + f_xf
        return net_torque

    def wheel_momentum_of_inertia(J_whf, J_fd, J_gb, i_tlc, J_m, i_hs):
        net_inertia = J_whf + J_fd + (J_gb * (i_tlc ** 2)) + (J_m * (i_tlc * i_hs) ** 2)
        return net_inertia

    net_torque = front_wheel_model(e_ft, f_bt, f_zf, f_xf, i_tlc, n_tlc, r_wh, i_df) 
    net_inertia = wheel_momentum_of_inertia(J_whf, J_fd, J_gb, i_tlc, J_m, i_hs)

    def integrate_angular_velocity(omega_prev, dt):
        angular_acceleration = net_torque / net_inertia
        omega_new = omega_prev + angular_acceleration * dt
        return omega_new

    wf = integrate_angular_velocity
    return wf
def front_omega_motor (wf,i_tlc,i_hs):
    ''' 
        wf: front wheel angular velocity function
        i_tlc, i_hs: gear ratios
    '''
    def omega_motor(omega_prev, dt):
        omega_wf = wf(omega_prev, dt)
        omega_motor_new = omega_wf * (i_tlc * i_hs)
        return omega_motor_new
    return omega_motor
def front_traction_forces_and_tier_slip(wf, v, r_wh):
    def slip_ratio_acceleration(wf, v, r_wh):
        TS = v - wf * r_wh
        MS = wf * r_wh
        if TS <= 0 and MS <= abs(TS)
            s1 = TS / MS
            return s1
        else:
            s1 = 0
            return s1
    def slip_ratio_braking(wf, v, r_wh):
        TS = v - wf * r_wh
        MS = v
        if TS > 0 and MS <= abs(TS):
            s2 = TS / MS
            return s2
        else:
            s2 = 0
            return s2

    s1 = slip_ratio_acceleration(wf, v, r_wh)
    s2 = slip_ratio_braking(wf, v, r_wh)
    if s1 == 0 and s2 == 0:
        return 0
    elif s1 <= 0:
        return 0 
    elif s2 >= 0:
        tire_slip = s2
        def friction_coefficient(tire_slip):
            file_path = "tiremodel.xls"
            data = pd.read_excel(file_path)  # không có tiêu đề

            Slip = data.iloc[:, 0].values   # cột 1
            Roadmu = data.iloc[:, 1].values # cột 2

            # Hàm tìm u(s) bằng nội suy tuyến tính
            def u_of_s(s):
                return np.interp(s, Slip, Roadmu)

            # Trả về hệ số ma sát cho tire_slip
            return u_of_s(tire_slip)
        f_xf = friction_coefficient(tire_slip)* f_zf
        return tire_slip, f_xf
