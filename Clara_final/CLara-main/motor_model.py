import numpy as np

class MotorModel:
    """
    3-phase PMSM (Permanent Magnet Synchronous Motor) Model
    Based on Simulink FRONT/REAR MOTOR block diagram
    
    Complete implementation including:
    - Tmax_rieng calculation (torque-speed characteristic)
    - Characteristic line Switch (torque limiting)
    - Choice Torque Switch1 (acceleration vs braking selection)
    - Power calculation for front and rear motors
    
    Inputs:
    - Front Motor Angular Velocity (rad/s)
    - Rear Motor Angular Velocity (rad/s)
    - Pedal Position
    - Front Required Regen Braking Torque (Nm) - from braking_model
    - Rear Required Regen Braking Torque (Nm) - from braking_model
    
    Outputs:
    - Front Motor Required Torque (no saturated) (Nm)
    - Rear Motor Required Torque (no saturated) (Nm)
    - Front Motor Required Power (kW)
    - Rear Motor Required Power (kW)
    """
    
    def __init__(self, max_motor_torque=520.0, omega_c=628.32):
        """
        Initialize Motor Model parameters
        
        Args:
            max_motor_torque (float): Maximum motor torque (Nm) - Tmax
            omega_c (float): Critical angular velocity (rad/s) - omega_c
        """
        self.Tmax = max_motor_torque  # Maximum motor torque (Nm)
        self.omega_c = omega_c  # Critical angular velocity (rad/s)
    """3phase PMSM Motor Model Methods"""
    def calculate_tmax_rieng(self,pedal_position):
        """
        Calculate Tmax_rieng based on motor angular velocity
        This implements the torque-speed characteristic curve
        From diagram: Tmax_rieng block
        """
        T_max_rieng = self.Tmax *  abs(pedal_position)
        return T_max_rieng
    
    def characteristic_line_switch(self, motor_angular_velocity, pedal_position, tmax_rieng):
        """
        Implements Characteristic line Switch logic from Simulink
        
        This switch selects between:
        - Upper path: Normal operation (|pedal_pos| * Tmax_rieng)
        - Lower path: Power-limited operation (moment_max / |omega|)
        
        Args:
            motor_angular_velocity (float): Motor angular velocity (rad/s)
            pedal_position (float): Pedal position (-1 to 1)
            tmax_rieng (float): Tmax_rieng value (Nm)
            
        Returns:
            float: Selected torque value (Nm)
        """
        omega_abs = abs(motor_angular_velocity)
        tmax_rieng = self.calculate_tmax_rieng(pedal_position)
        
        if omega_abs >= self.omega_c:
            # Power-limited operation
            # Calculate moment_max = omega_c * Tmax_rieng / omega
            moment_max = self.omega_c * tmax_rieng / omega_abs
            switch_output = moment_max
        else:
            switch_output = tmax_rieng
        return switch_output
    
    def choice_torque_switch(self, pedal_position, switch_output, regen_braking_torque):
        """
        Implements Choice Torque Switch1 logic from Simulink
        
        Selects between:
        - switch_output (acceleration mode, pedal >= 0)
        - Regen braking torque (braking mode, pedal < 0)
        
        Args:
            pedal_position (float): Pedal position (-1 to 1)
            switch_output (float): Torque from characteristic line switch (Nm)
            regen_braking_torque (float): Regen braking torque from braking_model (Nm)
            
        Returns:
            float: Selected torque (Nm)
        """
        # If pedal >= 0: acceleration mode, use switch_output
        # If pedal < 0: braking mode, use regen braking torque
        if pedal_position >= 0:
            return switch_output
        else:
            return abs(regen_braking_torque)
    
    def calculate_front_motor_torque(self, front_motor_angular_velocity, 
                                     pedal_position, 
                                     front_regen_torque):
        """
        Calculate Front Motor Required Torque (no saturated)
        
        Complete flow:
        1. Characteristic line Switch -> switch_output
        2. Choice Torque Switch1 -> if pedal >= 0: switch_output, else: regen_torque
        
        Args:
            front_motor_angular_velocity (float): Front motor angular velocity (rad/s)
            pedal_position (float): Pedal position (-1 to 1)
            front_regen_torque (float): Front regen braking torque (Nm)
            
        Returns:
            float: Front motor required torque (Nm)
        """
        # Step 1: Characteristic line Switch -> switch_output
        switch_output = self.characteristic_line_switch(
            front_motor_angular_velocity, 
            pedal_position, 
            None  # tmax_rieng is calculated inside
        )
        
        # Step 2: Choice Torque Switch1
        final_torque = self.choice_torque_switch(
            pedal_position, 
            switch_output * 0.5, 
            front_regen_torque
        )
        
        return final_torque
    
    def calculate_rear_motor_torque(self, rear_motor_angular_velocity, 
                                    pedal_position, 
                                    rear_regen_torque):
        """
        Calculate Rear Motor Required Torque (no saturated)
        
        Complete flow (same as front):
        1. Characteristic line Switch -> switch_output
        2. Choice Torque Switch1 -> if pedal >= 0: switch_output, else: regen_torque
        
        Args:
            rear_motor_angular_velocity (float): Rear motor angular velocity (rad/s)
            pedal_position (float): Pedal position (-1 to 1)
            rear_regen_torque (float): Rear regen braking torque (Nm)
            
        Returns:
            float: Rear motor required torque (Nm)
        """
        # Step 1: Characteristic line Switch -> switch_output
        switch_output = self.characteristic_line_switch(
            rear_motor_angular_velocity, 
            pedal_position, 
            None  # tmax_rieng is calculated inside
        )
        
        # Step 2: Choice Torque Switch1
        final_torque = self.choice_torque_switch(
            pedal_position, 
            switch_output * 1, 
            rear_regen_torque
        )
        
        return final_torque
        
    def calculate_motor_power(self, motor_angular_velocity, motor_torque):
        """
        Calculate motor power from angular velocity and torque
        Power(W) = Torque(Nm) * Angular_Velocity(rad/s)
        
        Args:
            motor_angular_velocity (float): Motor angular velocity (rad/s)
            motor_torque (float): Motor torque (Nm)
            
        Returns:
            float: Motor power in kW (with -K gain applied)
        """
        # Power = Torque * Omega
        power_watts = motor_torque * abs(motor_angular_velocity)
        
        # Convert to kW (removed -K gain to match battery model expectation of positive power for discharge)
        power_kw = power_watts / 1000.0
        
        return power_kw
    
    def get_outputs(self, front_motor_angular_velocity, 
                   rear_motor_angular_velocity,
                   pedal_position,
                   front_regen_torque,
                   rear_regen_torque):
        """
        Main function to get all motor model outputs
        
        Complete flow:
        1. Calculate final torque for front and rear (using characteristic line switch + choice torque switch)
        2. Calculate Power = final_torque × Omega × (-K)
        
        Args:
            front_motor_angular_velocity (float): Front motor angular velocity (rad/s)
            rear_motor_angular_velocity (float): Rear motor angular velocity (rad/s)
            pedal_position (float): Pedal position (-1 to 1)
            front_regen_torque (float): Front regen torque from braking_model (Nm)
            rear_regen_torque (float): Rear regen torque from braking_model (Nm)
            
        Returns:
            dict: Dictionary containing motor outputs
        """
        # Calculate front motor final torque
        front_final_torque = self.calculate_front_motor_torque(
            front_motor_angular_velocity,
            pedal_position,
            front_regen_torque
        )
        
        # Calculate rear motor final torque
        rear_final_torque = self.calculate_rear_motor_torque(
            rear_motor_angular_velocity,
            pedal_position,
            rear_regen_torque
        )
        
        # Calculate front motor required power using final torque
        front_required_power = self.calculate_motor_power(
            front_motor_angular_velocity,
            front_final_torque
        )
        
        # Calculate rear motor required power using final torque
        rear_required_power = self.calculate_motor_power(
            rear_motor_angular_velocity,
            rear_final_torque
        )
        
        return {
            'front_motor_required_torque': front_final_torque,     # Nm
            'rear_motor_required_torque': rear_final_torque,       # Nm
            'front_motor_required_power': front_required_power,    # kW
            'rear_motor_required_power': rear_required_power,      # kW
            'total_motor_power': front_required_power + rear_required_power  # kW
        }
