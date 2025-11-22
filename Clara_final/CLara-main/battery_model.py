import numpy as np

class BatteryModel:
    """
    Battery Model based on Simulink diagrams.
    Calculates Battery Voltage, Current, SOC, and Power Losses.
    """
    
    def __init__(self, 
                 initial_soc=0.95, 
                 capacity_ah=231.7, 
                 initial_voltage=350.0,
                 time_step=0.1,
                 max_motor_power_kw=250.0,
                 use_china_resistor=False):
        """
        Initialize Battery Model parameters
        
        Args:
            initial_soc (float): Initial State of Charge (0.0 to 1.0)
            capacity_ah (float): Battery Capacity in Amp-hours (used for K calculation)
            initial_voltage (float): Initial Battery Voltage (V)
            time_step (float): Simulation time step (s)
            max_motor_power_kw (float): Maximum Motor Power (kW) for HSC calculation
            use_china_resistor (bool): Switch to use 'Resistor from China' data
        """
        self.initial_soc = initial_soc
        self.soc = initial_soc
        self.capacity_ah = capacity_ah  # 231.7 in Simulink
        self.dt = time_step
        self.Pmax = max_motor_power_kw
        self.use_china_resistor = use_china_resistor
        
        # K factor for SOC calculation based on user request
        # K = 1 / (3600 * 231.7)
        self.k_soc = 1.0 / (3600.0 * self.capacity_ah)
        
        # Internal states
        self.current_voltage = initial_voltage
        self.current_current = 0.0  # Bat_Cur (Amps: Positive = Discharge, Negative = Charge)
        self.current_integral = 0.0  # Accumulator for Current Integral (∫ Bat_Cur dt)
        
        # Energy accumulators (matching Simulink blocks)
        self.total_energy_consumption_kwh = 0.0  # TotalConsum output
        self.energy_consumption_integral = 0.0  # Internal integrator for Total Energy
        self.total_regen_energy_kwh = 0.0  # ReEnergy output
        self.regen_power_integral = 0.0  # Internal integrator for Regen Energy
        self.total_regen_capacity_ah = 0.0  # Ah output (Regen Capacity)
        self.motor_power_loss_kwh = 0.0  # Phth (Motor Power Loss Energy)
        self.pinth_kw = 0.0  # pinth (Instantaneous Motor Power Loss)
        self.tpTHpin = 0.0 # User requested output
        self.tpTHpin_integral = 0.0 # Integral state for tpTHpin
        self.motor_power_loss_from_resistor_kwh = 0.0  # conhaoprrt (from Resistor)
        self.p_output_front_kw = 0.0
        self.p_output_rear_kw = 0.0
        self.limited_front_torque = 0.0
        self.limited_rear_torque = 0.0
        
        # Lookup Tables from Simulink (SOC to Battery Voltage and Internal Resistance)
        # SOC_V: SOC breakpoints in percentage
        self.SOC_V = np.array([
            13.709214501510580, 15.264350453172209, 16.378700906344413, 17.363293051359520, 
            18.451208459214502, 19.746223564954686, 21.377643504531722, 22.879456193353473, 
            24.329456193353479, 25.831117824773415, 27.643353474320243, 29.455438066465263, 
            31.293504531722061, 33.209214501510580, 35.409516616314207, 36.988670694864048, 
            39.603474320241695, 45.531570996978850, 53.789728096676740, 58.993051359516627, 
            64.274169184290031, 69.813897280966771, 74.318126888217520, 81.592296072507551, 
            84.854078549848950, 88.944410876132935, 92.050906344410890, 94.795015105740177, 
            96.425981873111795, 98.108610271903331, 99.273716012084591, 99.869184290030205
        ])
        
        # Vb: Battery Voltage values (Table data)
        self.Vb = np.array([
            3.076092000000000e+02, 3.373936000000000e+02, 3.578763000000000e+02, 3.712879000000000e+02, 
            3.824554000000000e+02, 3.920832000000000e+02, 4.018181000000000e+02, 4.086090000000000e+02, 
            4.139865000000000e+02, 4.180652000000000e+02, 4.206008000000000e+02, 4.217213000000000e+02, 
            4.228409000000000e+02, 4.234863000000000e+02, 4.245941000000000e+02, 4.260761000000000e+02, 
            4.284677000000000e+02, 4.324028000000000e+02, 4.367341000000000e+02, 4.391598000000000e+02, 
            4.415829000000000e+02, 4.436438000000000e+02, 4.451487000000000e+02, 4.475072000000000e+02, 
            4.490524000000000e+02, 4.523396000000000e+02, 4.541257000000000e+02, 4.560414000000000e+02, 
            4.575215000000000e+02, 4.594717000000000e+02, 4.614386000000000e+02, 4.636599000000000e+02
        ])
        
        # Internal Resistance Data (from Input_data_laptop (2).m)
        # SOC_Rin calculation: fliplr((66.2-[...])/66.2)*100
        raw_rin_offsets = np.array([6.6549, 13.2164, 19.8875, 26.4705, 39.7252, 46.3303, 52.9553])
        self.SOC_Rin = np.array([20.007099697885199, 30.014652567975830, 39.992145015105741, 60.014350453172206, 69.958459214501516, 80.035649546827798, 89.947280966767380])
        
        # Rint values (Ohm)
        self.Rint = np.array([0.202800000000000, 0.141800000000000, 0.139800000000000, 0.144600000000000, 0.139400000000000, 0.135800000000000, 0.145300000000000])
        
        # Resistor from China Data (from First Version/battery_models.py)
        # SOC_Rin_China = [0 ,20 ,40 ,60, 80] -> Added 100 to match Rint length
        self.SOC_Rin_China = np.array([0, 20, 40, 60, 80, 100])
        # Rint_China = [0.037, 0.036, 0.035, 0.0345, 0.034, 0.0348]
        self.Rint_China = np.array([0.037, 0.036, 0.035, 0.0345, 0.034, 0.0348])
        
        # Motor Efficiency Map Data (from Input_data_laptop (2).m)
        self.LoadRatio = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
        self.EffMt = np.array([58.72, 72.24, 83.89, 88.67, 91.28, 92.12, 92.71, 93.31, 93.91, 94.51, 94.43, 93.67, 92.91])
        self.Tmax = 520.0

    def interpolate_voltage_from_soc(self, soc_percent):
        """
        Interpolate Battery Voltage from SOC percentage
        Matches "Interpolate %SOC to U_bat" block in Simulink
        
        Args:
            soc_percent (float): SOC in percentage (0 to 100)
            
        Returns:
            float: Battery Voltage (V)
        """
        return np.interp(soc_percent, self.SOC_V, self.Vb)
    
    def interpolate_r_internal(self, soc_percent):
        """
        Interpolate Internal Resistance from SOC percentage
        Matches R_t lookup in Simulink with Switch for China Resistor
        
        Args:
            soc_percent (float): SOC in percentage (0 to 100)
            
        Returns:
            float: Internal Resistance (Ohm)
        """
        if self.use_china_resistor:
            return np.interp(soc_percent, self.SOC_Rin_China, self.Rint_China)
        else:
            return np.interp(soc_percent, self.SOC_Rin, self.Rint)
        
    def calculate_hsc(self, power_kw):
        """
        Calculate HSC (Efficiency Factor) based on Power
        Formula: HSC = Interp(abs(Power/Pmax), LoadRatio, EffMt) * 1/100
        
        Args:
            power_kw (float): Motor Power (kW)
            
        Returns:
            float: HSC factor
        """
        if self.Pmax == 0:
            return 0.0
            
        load_ratio = abs(power_kw / self.Pmax)  
        efficiency_percent = np.interp(load_ratio, self.LoadRatio, self.EffMt)
        return efficiency_percent / 100.0

    def calculate_current(self, pyc_w, battery_voltage, r_internal):
        """
        Calculate Battery Current based on Pyc (Power Yielded/Consumed)
        Matches Simulink MATLAB Function block 'fcn'
        
        Args:
            pyc_w (float): Total Power Consumed/Yielded (Watts)
            battery_voltage (float): Battery terminal voltage U_bat (V)
            r_internal (float): Internal resistance Rt (Ohm)
            
        Returns:
            float: Battery Current Bat_Cur (A)
        """
        # MATLAB logic translation:
        # function I_out = fcn(U,Rt,Pyc)
        # I_out=0;
        # a = Rt; b= -U; c= Pyc;
        
        I_out = 0.0
        a = r_internal
        b = -battery_voltage
        c = pyc_w
        
        # delta = b^2 - 4*a*c;
        delta = b**2 - 4*a*c
        
        # if delta >=0
        if delta >= 0:
            # I1 =  -(b + (b^2 - 4*a*c)^(1/2))/(2*a);
            I1 = -(b + np.sqrt(delta)) / (2*a)
            
            # I2 =  -(b - (b^2 - 4*a*c)^(1/2))/(2*a);
            I2 = -(b - np.sqrt(delta)) / (2*a)
            
            # if abs(I1)<abs(I2)
            if abs(I1) < abs(I2):
                I_out = I1
            else:
                I_out = I2
        
        # if delta <0
        else:
            I_out = 0.0
            
        return I_out

    def update_states(self, front_motor_power_kw, rear_motor_power_kw, 
                     front_motor_torque, rear_motor_torque, dt,
                     w_front=0, w_rear=0, velocity=0, acceleration=0,
                     front_traction_ratio=0.5, rear_traction_ratio=0.5):
        """
        Update battery states based on motor power demands
        Implements Simulink BATTERY subsystem logic
        
        Args:
            front_motor_power_kw (float): Front Motor Required Power (kW)
            rear_motor_power_kw (float): Rear Motor Required Power (kW)
            front_motor_torque (float): Front Motor Torque (Nm) (Requested)
            rear_motor_torque (float): Rear Motor Torque (Nm) (Requested)
            dt (float): Time step (s)
            w_front (float): Front motor angular velocity (rad/s)
            w_rear (float): Rear motor angular velocity (rad/s)
            velocity (float): Vehicle velocity (m/s)
            acceleration (float): Vehicle acceleration (m/s²)
            front_traction_ratio (float): Front traction ratio
            rear_traction_ratio (float): Rear traction ratio
            
        Returns:
            dict: Dictionary of battery outputs matching Simulink
        """
        self.dt = dt
        
        # Step 1: Calculate SOC
        # Formula from user: SOC = Initial_SOC - Integral(Bat_Cur * K)
        # where K = 1/(3600 * 231.7)
        
        # Update integral of current: ∫(Bat_Cur) dt
        self.current_integral += self.current_current * dt
        
        # Calculate SOC (fraction 0-1)
        self.soc = self.initial_soc - (self.current_integral * self.k_soc)
        
        # Clamp SOC to valid range
        self.soc = max(0.0, min(1.0, self.soc))
        soc_percent = self.soc * 100.0
        
        # Step 2: Interpolate Battery Voltage from SOC
        battery_voltage = self.interpolate_voltage_from_soc(soc_percent)
        
        # Step 3: Get Internal Resistance from SOC
        r_internal = self.interpolate_r_internal(soc_percent)
        
        # Step 4: Calculate Pyc (Power Yielded/Consumed)
        # Calculate HSC factors
        hsc_front = self.calculate_hsc(front_motor_power_kw)
        hsc_rear = self.calculate_hsc(rear_motor_power_kw)
        
        # Calculate Pyc_front and Pyc_rear (Watts)
        # Pyc_front = FRP(Watts) * HSC_front
        pyc_front_w = (front_motor_power_kw * 1000.0) / hsc_front if hsc_front != 0 else 0
        pyc_rear_w = (rear_motor_power_kw * 1000.0) / hsc_rear if hsc_rear != 0 else 0
        
        # Total Pyc (Watts)
        self.pyc_w = pyc_front_w + pyc_rear_w
        
        # Calculate PoutputFront/Rear (kW) for reporting
        self.p_output_front_kw = front_motor_power_kw * hsc_front
        self.p_output_rear_kw = rear_motor_power_kw * hsc_rear
        
        # Step 5: Calculate Battery Current
        # Uses fcn block with inputs: U, Rt, Pyc
        self.current_current = self.calculate_current(
            self.pyc_w, 
            battery_voltage, 
            r_internal
        )
        
        # Step 6: Calculate Terminal Voltage
        # U_terminal = U_bat - I * Rt
        self.current_voltage = battery_voltage - self.current_current * r_internal
        
        # Step 7: Calculate Motor Power Loss
        # Motor Power Loss (pinth) = I^2 * Rt
        current_squared = self.current_current ** 2
        motor_power_loss_w = current_squared * r_internal
        self.pinth_kw = motor_power_loss_w / 1000.0  # Instantaneous Power Loss (kW)
        self.motor_power_loss_kwh += self.pinth_kw * (dt / 3600.0)
        
        # Calculate tpTHpin as requested: pinth_kw * 1/3600 -> Integral -> * 1/1000
        val_to_integrate_tp = self.pinth_kw * (1.0/3600.0)
        self.tpTHpin_integral += val_to_integrate_tp * dt
        self.tpTHpin = self.tpTHpin_integral / 1000.0
        
        # Motor Power Loss from Resistor (tonhaomotor)
        # Formula: p_output_front_kw + p_output_rear_kw
        tonhaomotor_kw = self.p_output_front_kw + self.p_output_rear_kw
        self.motor_power_loss_from_resistor_kwh = tonhaomotor_kw
        
        # Step 8: Calculate Total Energy Consumption
        # Formula: Integral( I*V * 1/3600 ) * 1/1000
        
        # 1. Calculate Power (Watts)
        power_watts_consumption = self.current_current * self.current_voltage
        
        # 2. Multiply by 1/3600
        val_to_integrate_consumption = power_watts_consumption / 3600.0
        
        # 3. Integrate
        self.energy_consumption_integral += val_to_integrate_consumption * dt
        
        # 4. Multiply by 1/1000
        self.total_energy_consumption_kwh = self.energy_consumption_integral / 1000.0
        
        # Step 9: Calculate Regenerative Energy
        # Formula: abs( Integral( min(0, I*V) * 1/3600 ) * 1/1000 )
        
        # 1. Calculate Power (Watts)
        power_watts = self.current_current * self.current_voltage
        
        # 2. Keep only negative values (Saturation Upper Limit 0)
        saturated_power = min(0.0, power_watts)
        
        # 3. Multiply by 1/3600
        val_to_integrate = saturated_power / 3600.0
        
        # 4. Integrate
        self.regen_power_integral += val_to_integrate * dt
        
        # 5. Multiply by 1/1000 and take Absolute value
        self.total_regen_energy_kwh = abs(self.regen_power_integral / 1000.0)
        
        # Step 10: Calculate Regen Capacity in Ah
        # Formula: regen_energy / (battery_voltage * 1000)
        if self.current_voltage != 0:
            self.total_regen_capacity_ah = self.total_regen_energy_kwh / (self.current_voltage * 1000.0)
        else:
            self.total_regen_capacity_ah = 0.0
            
        # Step 11: Calculate Limited Motor Torque (Feedback to Tire Model)
        # Logic updated based on user request
        
        # Calculate Battery Power (Watts) - Terminal Power
        battery_power_watts = self.current_current * self.current_voltage
        
        # Calculate Internal Loss (Watts)
        pinth_watts = self.pinth_kw * 1000.0
        
        # Available power for motors (User formula: battery_power - pinth)
        available_power_watts = battery_power_watts - pinth_watts
        
        # Front Calculation
        # A_P (Available Power)
        A_P_front = available_power_watts * hsc_front * front_traction_ratio
        
        # R_P (Requested Power)
        R_P_front = self.pyc_w * front_traction_ratio
        
        # R_T (Requested Torque)
        R_T_front = front_motor_torque
        
        # A_T (Available Torque)
        if w_front != 0:
            A_T_front = A_P_front / w_front
        else:
            A_T_front = 0.0
            
        # Check Power Logic
        torque_front_raw = self.check_power_logic(R_P_front, A_P_front, R_T_front, A_T_front, velocity, acceleration)
        
        # Saturation -320 to 320
        self.limited_front_torque = max(-320.0, min(320.0, torque_front_raw))
        
        # Rear Calculation (Assuming symmetric logic)
        A_P_rear = available_power_watts * hsc_rear * rear_traction_ratio
        R_P_rear = self.pyc_w * rear_traction_ratio
        R_T_rear = rear_motor_torque
        
        if w_rear != 0:
            A_T_rear = A_P_rear / w_rear
        else:
            A_T_rear = 0.0
            
        torque_rear_raw = self.check_power_logic(R_P_rear, A_P_rear, R_T_rear, A_T_rear, velocity, acceleration)
        
        # Saturation -320 to 320
        self.limited_rear_torque = max(-320.0, min(320.0, torque_rear_raw))
        
        return self.get_outputs()

    def check_power_logic(self, R_P, A_P, R_T, A_T, v, a):
        """
        Implements the check power logic described by user
        function [Torque,check] = fcn(R_P, A_P, R_T,A_T,v,a)
        """
        Torque = 0.0
        
        if A_P < R_P:
            Torque = A_T
        else:
            Torque = R_T
            
        if R_P < 0:
            Torque = R_T
            
        if v < 0.5 and a < 0:
            Torque = 0.0
            
        return Torque

    def get_outputs(self):
        """
        Return current state of the battery
        Matches Simulink output ports
        """
        soc_percent = self.soc * 100.0
        return {
            # Main battery states
            'soc': self.soc,  # SOC as fraction (0-1)
            'soc_percent': soc_percent,  # SOC as percentage (0-100)
            'voltage': self.current_voltage,  # Bat_Volt - Terminal voltage (V)
            'current': self.current_current,  # Bat_Cur - Battery current (A)
            'power_watts': self.current_current * self.current_voltage,  # Instantaneous Power (W)
            'battery_voltage_ocv': self.interpolate_voltage_from_soc(soc_percent),  # U_bat from lookup
            'internal_resistance': self.interpolate_r_internal(soc_percent),  # Rt (Ohm)
            
            # Energy metrics (matching Simulink outputs)
            'total_energy_consumption_kwh': self.total_energy_consumption_kwh,  # TotalConsum
            'total_regen_energy_kwh': self.total_regen_energy_kwh,  # ReEnergy
            'total_regen_capacity_ah': self.total_regen_capacity_ah,  # Ah (Regen Capacity)
            
            # Power loss metrics
            'motor_power_loss_kwh': self.motor_power_loss_kwh,  # Accumulated Energy Loss
            'pinth_kw': self.pinth_kw,  # Instantaneous Power Loss (pinth)
            'tpTHpin': self.tpTHpin, # User requested output
            'motor_power_loss_from_resistor_kwh': self.motor_power_loss_from_resistor_kwh,  # conhaoprrt
            'current_squared': self.current_current ** 2,  # u(1)^2 output
            
            # Power outputs (HSC * Power)
            'p_output_front_kw': self.p_output_front_kw,
            'p_output_rear_kw': self.p_output_rear_kw,
            
            # Limited Torques (Feedback to Tire Model)
            'limited_front_torque': self.limited_front_torque,
            'limited_rear_torque': self.limited_rear_torque
        }

# Example usage
if __name__ == "__main__":
    battery = BatteryModel()
    print("Testing Battery Model...")
    
    # Test Discharge
    print("\n--- Discharge Test (50kW Load) ---")
    outputs = battery.update_states(30.0, 20.0, 1.0) # 50kW for 1 sec
    print(f"Voltage: {outputs['voltage']:.2f} V")
    print(f"Current: {outputs['current']:.2f} A")
    print(f"SOC: {outputs['soc']:.6f}")
    
    # Test Regen
    print("\n--- Regen Test (-20kW Load) ---")
    outputs = battery.update_states(-10.0, -10.0, 1.0) # -20kW for 1 sec
    print(f"Voltage: {outputs['voltage']:.2f} V")
    print(f"Current: {outputs['current']:.2f} A")
    print(f"SOC: {outputs['soc']:.6f}")
    print(f"Regen Energy: {outputs['total_regen_energy_kwh']:.6f} kWh")
