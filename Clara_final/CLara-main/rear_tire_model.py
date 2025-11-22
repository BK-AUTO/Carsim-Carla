import numpy as np
import math
class RearTireModel:
    """
    Rear Tire Model based on Simulink diagram
    Calculates tire dynamics, slip, traction force, and vertical load
    Same structure as front tire model but for rear wheels
    """
    
    def __init__(self, vehicle_mass, wheelbase_length, cg_height, 
                 tire_radius, gear_ratio_tlc, gear_ratio_hs,
                 inertia_wheel, inertia_diff, inertia_gearbox, 
                 inertia_motor, tire_stiffness_c1=None, tire_stiffness_c2=None, 
                 tire_stiffness_c3=None,n_hs=0.95, n_tlc=0.95,id_f = 0.016):
        """
        Initialize rear tire model parameters
        
        Args:
            vehicle_mass (float): Vehicle mass (m) in kg
            wheelbase_length (float): Wheelbase length (l) in m
            cg_height (float): Center of gravity height (h) in m
            tire_radius (float): Tire radius (r_wh) in m
            gear_ratio_tlc (float): Gear ratio i_tlc
            gear_ratio_hs (float): Gear ratio i_hs
            inertia_wheel (float): Wheel inertia (Jwh) in kg*m²
            inertia_diff (float): Differential inertia (Jfd) in kg*m²
            inertia_gearbox (float): Gearbox inertia (Jgb) in kg*m²
            inertia_motor (float): Motor inertia (Jm) in kg*m²
            tire_stiffness_c1, c2, c3 (float): Tire stiffness parameters for slip calculation
        """
        self.m = vehicle_mass
        self.l = wheelbase_length
        self.h = cg_height
        self.r_wh = tire_radius
        self.i_tlc = gear_ratio_tlc
        self.i_hs = gear_ratio_hs
        self.g = 9.81  # Gravitational acceleration (m/s²)
        self.n_hs = n_hs  # High speed gear ratio multiplier
        self.n_tlc = n_tlc  # Transmission/differential gear ratio multiplier
        self.id_f = id_f  # Drag coefficient or similar factor
        
        # Inertias
        self.Jwh = inertia_wheel
        self.Jfd = inertia_diff
        self.Jgb = inertia_gearbox
        self.Jm = inertia_motor
        
        # Tire stiffness parameters (for slip ratio calculation)
        self.c1 = tire_stiffness_c1 if tire_stiffness_c1 is not None else 1.0
        self.c2 = tire_stiffness_c2 if tire_stiffness_c2 is not None else 1.0
        self.c3 = tire_stiffness_c3 if tire_stiffness_c3 is not None else 1.0
        
        # Internal states
        self.rear_tire_angular_velocity = 0.0  # rad/s
        
    def calculate_rear_vertical_load(self, vehicle_acceleration):
        """
        Calculate rear tire vertical load (F_zr) based on Simulink diagram
        Formula: Fzr = (g*lf - h*a) * m/l
        
        Args:
            vehicle_acceleration (float): Vehicle acceleration in m/s²
            
        Returns:
            float: Rear tire vertical load in N
        """
        # From the Simulink block: (g*lf - h*acceleration) * m/l
        # Assuming lf (front wheelbase distance) = l/2 for simplification
        # More accurate would be to pass lf as a parameter
        lf = self.l / 2.0  # Simplified assumption
        
        fzr = (self.g * lf + self.h * vehicle_acceleration) * self.m / self.l
        return fzr
    
    def calculate_rear_motor_angular_velocity(self, rear_tire_angular_velocity):
        """
        Calculate rear motor angular velocity from tire angular velocity
        Formula: w_motor = w_tire * i_tlc * i_hs
        
        Args:
            rear_tire_angular_velocity (float): Rear tire angular velocity in rad/s
            
        Returns:
            float: Rear motor angular velocity in rad/s
        """
        w_motor = rear_tire_angular_velocity * self.i_tlc * self.i_hs
        return w_motor
    
    def calculate_tire_slip(self, rear_tire_angular_velocity, vehicle_velocity_ms):
        """
        Calculate tire slip ratio based on Simulink diagram
        Uses two calculation blocks and selects appropriate one
        
        Args:
            rear_tire_angular_velocity (float): Rear tire angular velocity in rad/s
            vehicle_velocity_ms (float): Vehicle velocity in m/s
            
        Returns:
            float: Tire slip ratio (dimensionless)
        """
        # Calculate slip ratio 1: (r_wh * w - v) / v
        if abs(vehicle_velocity_ms) > 0.01:  # Avoid division by zero
            slip_ratio1 = (self.r_wh * rear_tire_angular_velocity - vehicle_velocity_ms) / vehicle_velocity_ms
        else:
            slip_ratio1 = 0.0
        
        # Calculate slip ratio 2: (r_wh * w - v) / (r_wh * w)
        wheel_velocity = self.r_wh * rear_tire_angular_velocity
        if abs(wheel_velocity) > 0.01:  # Avoid division by zero
            slip_ratio2 = (wheel_velocity - vehicle_velocity_ms) / wheel_velocity
        else:
            slip_ratio2 = 0.0
        
        # Select appropriate slip ratio based on conditions (similar to Simulink logic)
        # Typically use slip_ratio1 when vehicle is moving, slip_ratio2 when wheel is spinning
        if abs(vehicle_velocity_ms) > abs(wheel_velocity):
            slip_ratio = slip_ratio1
        else:
            slip_ratio = slip_ratio2
        
        # Apply saturation/limits as shown in Simulink (phiMax block)
        # Limit slip ratio to reasonable range
        slip_ratio = np.clip(slip_ratio, -1.0, 1.0)
        
        return slip_ratio
    
    def calculate_traction_force(self, slip_ratio, rear_vertical_load):
        """
        Calculate rear tire traction force based on slip ratio and vertical load
        From Simulink: Uses Lookup table for interpolation-extrapolation between slip and road mu
        From tiremodels.xls: left column is slip, right column is roadmu
        
        Args:
            slip_ratio (float): Tire slip ratio
            rear_vertical_load (float): Rear vertical load in N
            
        Returns:
            float: Rear tire traction force in N
        """
        # Tire model lookup table from tiremodels.xls
        # Left column: slip ratio, Right column: road mu (friction coefficient)
        slip_table = np.array([
            -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ])
        
        roadmu_table = np.array([
            -0.8, -0.78, -0.75, -0.72, -0.68, -0.63, -0.57, -0.5, -0.42, -0.33,
            0.0, 0.33, 0.42, 0.5, 0.57, 0.63, 0.68, 0.72, 0.75, 0.78, 0.8
        ])
        
        # Perform 1-D linear interpolation with extrapolation
        # This matches the Simulink Lookup block behavior
        traction_coefficient = np.interp(slip_ratio, slip_table, roadmu_table)
        
        # Calculate traction force: Fx = mu * Fz
        traction_force = traction_coefficient *0.9 * rear_vertical_load
        
        # Apply saturation to prevent unrealistic forces
        max_traction = rear_vertical_load * 1.0  # Maximum mu ~ 1.0
        traction_force = np.clip(traction_force, -max_traction, max_traction)
        
        return traction_force
    
    def calculate_tire_angular_acceleration(self, rear_motor_torque, rear_mechanical_braking_torque,
                                           rear_traction_force, rear_vertical_load):
        """
        Calculate rear tire angular acceleration based on torque balance
        From Simulink Rear Tire Model block:
        Formula: alpha = -K * (T_motor*i_tlc*i_tlc - T_brake - F_traction*r_wh - F_vertical*r_wh)
        where K = 1/(Jwh + Jfd + Jgb*i_tlc^2 + Jm*i_tlc^2*i_hs^2)
        
        Args:
            rear_motor_torque (float): Rear motor torque in Nm
            rear_mechanical_braking_torque (float): Mechanical braking torque in Nm
            rear_traction_force (float): Rear tire traction force in N
            rear_vertical_load (float): Rear tire vertical load in N
            
        Returns:
            float: Rear tire angular acceleration in rad/s²
        """
        # Calculate equivalent inertia (denominator in K)
        total_inertia = (self.Jwh + self.Jfd + 
                        self.Jgb * (self.i_tlc ** 2) + 
                        self.Jm * (self.i_tlc ** 2) * (self.i_hs ** 2))
        
        # Calculate K coefficient
        K = 1.0 / total_inertia
        
        # Calculate torque balance
        # Motor torque contribution (through gear ratio)
        motor_torque_contribution = rear_motor_torque * self.i_tlc * self.n_tlc * self.i_hs * self.n_hs
        
        # Traction force torque (opposing motion)
        traction_torque = rear_traction_force * self.r_wh
        
        # Vertical load contribution (appears as r_wh1 in diagram, might be drag)
        # This seems unusual but follows the Simulink diagram structure
        vertical_torque = rear_vertical_load * self.r_wh * self.id_f # This might need clarification
        
        # Total torque balance (note the signs from Simulink)
        net_torque = motor_torque_contribution - rear_mechanical_braking_torque - traction_torque - vertical_torque
        
        # Calculate angular acceleration
        # Corrected sign: alpha = net_torque / Inertia
        angular_acceleration = K * net_torque
        
        return angular_acceleration
    
    def update_tire_angular_velocity(self, angular_acceleration, dt):
        """
        Update tire angular velocity using integration
        
        Args:
            angular_acceleration (float): Angular acceleration in rad/s²
            dt (float): Time step in seconds
            
        Returns:
            float: Updated rear tire angular velocity in rad/s
        """
        self.rear_tire_angular_velocity += angular_acceleration * dt
        
        # Prevent negative angular velocity (wheel can't spin backwards in this simplified model)
        if self.rear_tire_angular_velocity < 0:
            self.rear_tire_angular_velocity = 0
        
        return self.rear_tire_angular_velocity
    
    def get_outputs(self, vehicle_acceleration, vehicle_velocity_ms, 
                   rear_motor_torque, rear_mechanical_braking_torque, 
                   current_time, dt=0.1):
        """
        Main function to calculate all rear tire outputs
        
        Args:
            vehicle_acceleration (float): Vehicle acceleration in m/s²
            vehicle_velocity_ms (float): Vehicle velocity in m/s
            rear_motor_torque (float): Rear motor torque in Nm
            rear_mechanical_braking_torque (float): Mechanical braking torque in Nm
            current_time (float): Current simulation time in seconds
            dt (float): Time step in seconds
            
        Returns:
            dict: Dictionary containing all rear tire outputs
        """
        # 1. Calculate rear vertical load
        rear_vertical_load = self.calculate_rear_vertical_load(vehicle_acceleration)
        
        # 2. Calculate tire slip
        tire_slip = self.calculate_tire_slip(self.rear_tire_angular_velocity, vehicle_velocity_ms)
        
        # 3. Calculate traction force
        rear_traction_force = self.calculate_traction_force(tire_slip, rear_vertical_load)
        
        # 4. Calculate tire angular acceleration
        tire_angular_acceleration = self.calculate_tire_angular_acceleration(
            rear_motor_torque, rear_mechanical_braking_torque,
            rear_traction_force, rear_vertical_load
        )
        
        # 5. Update tire angular velocity
        rear_tire_angular_velocity = self.update_tire_angular_velocity(tire_angular_acceleration, dt)
        
        # 6. Calculate rear motor angular velocity
        rear_motor_angular_velocity = self.calculate_rear_motor_angular_velocity(rear_tire_angular_velocity)
        
        return {
            'rear_vertical_load': rear_vertical_load,  # N
            'rear_tire_slip': tire_slip,  # dimensionless
            'rear_traction_force': rear_traction_force,  # N
            'rear_tire_angular_velocity': rear_tire_angular_velocity,  # rad/s
            'rear_tire_angular_acceleration': tire_angular_acceleration,  # rad/s²
            'rear_motor_angular_velocity': rear_motor_angular_velocity,  # rad/s
            'time': current_time
        }
    
    def reset_states(self):
        """Reset all internal states"""
        self.rear_tire_angular_velocity = 0.0


# Example usage and testing
if __name__ == "__main__":
    # Create rear tire model instance with typical values
    rear_tire = RearTireModel(
        vehicle_mass=2530,          # kg
        wheelbase_length=2.875,     # m
        cg_height=0.540,            # m
        tire_radius=0.334,          # m (typical for 225/55R17)
        gear_ratio_tlc=9.73,        # Transmission/differential gear ratio
        gear_ratio_hs=1.0,          # High speed gear ratio
        inertia_wheel=1.5,          # kg*m²
        inertia_diff=0.05,          # kg*m²
        inertia_gearbox=0.02,       # kg*m²
        inertia_motor=0.1           # kg*m²
    )
    
    print("Testing Rear Tire Model...")
    print("=" * 80)
    
    # Simulation parameters
    dt = 0.1  # Time step (seconds)
    total_time = 10.0  # Total simulation time
    time_steps = np.arange(0, total_time + dt, dt)
    
    # Test scenario: acceleration from standstill
    print("\nScenario: Acceleration from standstill")
    print("-" * 80)
    print(f"{'Time':<6} {'Fzr':<10} {'Slip':<10} {'Fxr':<10} {'w_tire':<12} {'w_motor':<12}")
    print(f"{'(s)':<6} {'(N)':<10} {'':<10} {'(N)':<10} {'(rad/s)':<12} {'(rad/s)':<12}")
    print("-" * 80)
    
    # Initial conditions
    vehicle_velocity = 0.0  # m/s
    vehicle_acceleration = 1.5  # m/s² (moderate acceleration)
    motor_torque = 200.0  # Nm
    braking_torque = 0.0  # Nm
    
    for t in time_steps:
        # Get outputs
        outputs = rear_tire.get_outputs(
            vehicle_acceleration,
            vehicle_velocity,
            motor_torque,
            braking_torque,
            t,
            dt
        )
        
        # Update vehicle velocity (simplified - would come from longitudinal model)
        vehicle_velocity += vehicle_acceleration * dt
        
        # Reduce acceleration as speed increases (simplified)
        if vehicle_velocity > 10:
            vehicle_acceleration = 0.5
            motor_torque = 100.0
        
        # Print every second
        if abs(t % 1.0) < dt/2:
            print(f"{t:<6.1f} {outputs['rear_vertical_load']:<10.1f} "
                  f"{outputs['rear_tire_slip']:<10.4f} {outputs['rear_traction_force']:<10.1f} "
                  f"{outputs['rear_tire_angular_velocity']:<12.2f} "
                  f"{outputs['rear_motor_angular_velocity']:<12.2f}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    
    # Print final state
    print(f"\nFinal rear tire angular velocity: {rear_tire.rear_tire_angular_velocity:.2f} rad/s")
    print(f"Final vehicle velocity: {vehicle_velocity:.2f} m/s ({vehicle_velocity*3.6:.1f} km/h)")