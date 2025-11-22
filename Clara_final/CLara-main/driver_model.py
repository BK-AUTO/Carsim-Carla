import numpy as np
import pandas as pd

class DriverModel:
    """
    Driver Model implementing NEDC driving cycle with PID control
    """
    
    def __init__(self):
        # PID Controller parameters (matching Simulink block)
        self.Kp = 0.6   # Proportional gain
        self.Ki = 0.0   # Integral gain (disabled in Simulink)
        self.Kd = 0.15  # Derivative gain
        self.N = 30     # Filter coefficient for derivative term
        
        # PID internal states
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0
        self.filtered_derivative = 0.0  # For filtered derivative term
        
        # NEDC driving cycle data (time in seconds, velocity in km/h)
        self.nedc_data = self._load_nedc_cycle()
        
        # Vehicle parameters
        self.dt = 0.1  # Time step for numerical differentiation
        
    def _load_nedc_cycle(self):
        """
        Load NEDC (New European Driving Cycle) velocity profile
        Returns array of [time, velocity_kmh] pairs
        """
        # NEDC cycle data: time(s) vs velocity(km/h)
        # This is the standard NEDC profile used in vehicle testing
        time_points = np.array([
            0, 11, 15, 23, 28, 49, 54, 56, 61, 85, 93, 96, 117, 143, 155, 163, 176, 185, 195,
            205, 220, 235, 251, 267, 282, 298, 314, 329, 345, 361, 376, 392, 407, 423, 439, 
            454, 470, 486, 501, 517, 533, 548, 564, 579, 595, 611, 626, 642, 658, 673, 689,
            704, 720, 736, 751, 767, 783, 798, 814, 829, 845, 861, 876, 892, 908, 923, 939,
            954, 970, 986, 1001, 1017, 1033, 1048, 1064, 1079, 1095, 1111, 1126, 1142, 1158, 1180
        ])
        
        velocity_kmh = np.array([
            0, 0, 15, 15, 10, 10, 0, 0, 15, 15, 15, 32, 32, 32, 50, 50, 50, 35, 35,
            35, 35, 50, 50, 50, 50, 50, 50, 35, 35, 35, 35, 50, 50, 50, 50, 50, 50,
            35, 35, 35, 35, 50, 50, 50, 50, 50, 50, 35, 35, 35, 35, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ])
        
        return np.column_stack((time_points, velocity_kmh))
    
    def get_desired_velocity(self, current_time):
        """
        Get desired velocity from NEDC cycle at given time
        
        Args:
            current_time (float): Current simulation time in seconds
            
        Returns:
            float: Desired velocity in km/h
        """
        # Handle time beyond NEDC cycle
        if current_time >= self.nedc_data[-1, 0]:
            return 0.0
        
        # Interpolate velocity from NEDC data
        desired_velocity_kmh = np.interp(current_time, self.nedc_data[:, 0], self.nedc_data[:, 1])
        return desired_velocity_kmh
    
    def kmh_to_ms(self, velocity_kmh):
        """
        Convert velocity from km/h to m/s
        
        Args:
            velocity_kmh (float): Velocity in km/h
            
        Returns:
            float: Velocity in m/s
        """
        return velocity_kmh / 3.6
    
    def ms_to_kmh(self, velocity_ms):
        """
        Convert velocity from m/s to km/h
        
        Args:
            velocity_ms (float): Velocity in m/s
            
        Returns:
            float: Velocity in km/h
        """
        return velocity_ms * 3.6
    
    def calculate_desired_acceleration(self, current_time, current_velocity_ms):
        """
        Calculate desired acceleration based on desired velocity profile
        
        Args:
            current_time (float): Current simulation time in seconds
            current_velocity_ms (float): Current vehicle velocity in m/s
            
        Returns:
            float: Desired acceleration in m/s²
        """
        # Get desired velocity at current time and slightly ahead
        dt = 0.1  # Small time step for numerical differentiation
        
        desired_vel_now_kmh = self.get_desired_velocity(current_time)
        desired_vel_next_kmh = self.get_desired_velocity(current_time + dt)
        
        # Convert to m/s
        desired_vel_now_ms = self.kmh_to_ms(desired_vel_now_kmh)
        desired_vel_next_ms = self.kmh_to_ms(desired_vel_next_kmh)
        
        # Calculate desired acceleration
        desired_acceleration = (desired_vel_next_ms - desired_vel_now_ms) / dt
        
        return desired_acceleration
    
    def calculate_desired_distance(self, current_time):
        """
        Calculate desired distance traveled based on NEDC profile
        
        Args:
            current_time (float): Current simulation time in seconds
            
        Returns:
            float: Desired distance in meters
        """
        if current_time <= 0:
            return 0.0
        
        # Integrate velocity over time to get distance
        time_points = np.arange(0, min(current_time, self.nedc_data[-1, 0]), self.dt)
        
        desired_distance = 0.0
        for t in time_points:
            vel_kmh = self.get_desired_velocity(t)
            vel_ms = self.kmh_to_ms(vel_kmh)
            desired_distance += vel_ms * self.dt
            
        return desired_distance
    
    def pid_controller(self, desired_velocity_ms, actual_velocity_ms, current_time):
    
        """
        PID controller to calculate pedal position based on velocity error
        Implements parallel form with filtered derivative as in Simulink
        
        Args:
            desired_velocity_ms (float): Desired velocity in m/s
            actual_velocity_ms (float): Actual vehicle velocity in m/s
            current_time (float): Current simulation time in seconds
            
        Returns:
            float: Pedal position (0.0 to 1.0 for acceleration, -1.0 to 0.0 for braking)
        """
        # Calculate velocity error
        error = desired_velocity_ms - actual_velocity_ms
        
        # Calculate time step
        if self.previous_time == 0:
            dt = 0.1
        else:
            dt = current_time - self.previous_time
        
        if dt <= 0:
            dt = 0.1
        
        # Proportional term
        proportional = self.Kp * error
        
        # Integral term (disabled as Ki = 0 in Simulink)
        if self.Ki != 0:
            self.integral_error += error * dt
            integral = self.Ki * self.integral_error
        else:
            integral = 0.0
        
        # Filtered Derivative term (as per Simulink PID block)
        # Using first-order filter: D_filtered = N*D/(s + N)
        # Discrete approximation: D_filtered(k) = (N*dt)/(N*dt + 1) * D_raw + (1)/(N*dt + 1) * D_filtered(k-1)
        if dt > 0:
            raw_derivative = (error - self.previous_error) / dt
            alpha = (self.N * dt) / (self.N * dt + 1)
            self.filtered_derivative = alpha * raw_derivative + (1 - alpha) * self.filtered_derivative
            derivative = self.Kd * self.filtered_derivative
        else:
            derivative = 0.0
        
        # Calculate PID output (Parallel form: P + I + D)
        pid_output = proportional + integral + derivative
        
        # Update previous values
        self.previous_error = error
        self.previous_time = current_time
        
        # Convert PID output to pedal position
        # Positive values = acceleration pedal (0 to 1)
        # Negative values = brake pedal (-1 to 0)
        pedal_position = np.clip(pid_output, -1.0, 1.0)
        
        # Set pedal to neutral when desired velocity is very low
        if abs(pedal_position) < 0.01 and desired_velocity_ms < 0.05:
            pedal_position = 0.0
            
        return pedal_position
    
    def get_outputs(self, current_time, actual_velocity_ms):
        """
        Main function to get all driver model outputs
        
        Args:
            current_time (float): Current simulation time in seconds
            actual_velocity_ms (float): Actual vehicle velocity in m/s
            
        Returns:
            dict: Dictionary containing all outputs
                - desired_velocity_kmh: Desired velocity in km/h
                - desired_velocity_ms: Desired velocity in m/s
                - desired_acceleration: Desired acceleration in m/s²
                - desired_distance: Desired distance in meters
                - pedal_position: Pedal position (-1.0 to 1.0)
                - velocity_error: Velocity error in m/s
        """
        # Get desired velocity
        desired_velocity_kmh = self.get_desired_velocity(current_time)
        desired_velocity_ms = self.kmh_to_ms(desired_velocity_kmh)
        
        # Calculate desired acceleration
        desired_acceleration = self.calculate_desired_acceleration(current_time, actual_velocity_ms)
        
        # Calculate desired distance
        desired_distance = self.calculate_desired_distance(current_time)
        
        # Calculate pedal position using PID controller
        pedal_position = self.pid_controller(desired_velocity_ms, actual_velocity_ms, current_time)
        
        # Calculate velocity error
        velocity_error = desired_velocity_ms - actual_velocity_ms
        
        return {
            'desired_velocity_kmh': desired_velocity_kmh,
            'desired_velocity_ms': desired_velocity_ms,
            'desired_acceleration': desired_acceleration,
            'desired_distance': desired_distance,
            'pedal_position': pedal_position,
            'velocity_error': velocity_error
        }
    

    
    def reset_pid(self):
        """Reset PID controller internal states"""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0
        self.filtered_derivative = 0.0
    
    def set_pid_gains(self, kp, ki, kd, n=30):
        """
        Set PID controller gains
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            n (float): Filter coefficient for derivative term
        """
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.N = n
    
    def plot_nedc_cycle(self):
        """
        Plot the NEDC driving cycle for visualization
        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.nedc_data[:, 0], self.nedc_data[:, 1], 'b-', linewidth=2)
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (km/h)')
            plt.title('NEDC (New European Driving Cycle) Velocity Profile')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1200)
            plt.ylim(0, max(self.nedc_data[:, 1]) + 5)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot NEDC cycle.")


# Example usage and testing
if __name__ == "__main__":
    # Create driver model instance
    driver = DriverModel()
    
    # Test the model
    print("Testing Driver Model...")
    print("-" * 50)
    
    # Test at different time points
    test_times = [0, 50, 100, 200, 500, 800, 1000, 1180]
    current_velocity_ms = 0.0  # Assume vehicle is moving at 5 m/s (18 km/h)
    
    for t in test_times:
        outputs = driver.get_outputs(t, current_velocity_ms)
        
        print(f"Time: {t:4.0f}s")
        print(f"  Desired Velocity: {outputs['desired_velocity_kmh']:6.1f} km/h ({outputs['desired_velocity_ms']:6.2f} m/s)")
        print(f"  Desired Acceleration: {outputs['desired_acceleration']:8.3f} m/s²")
        print(f"  Desired Distance: {outputs['desired_distance']:8.1f} m")
        print(f"  Pedal Position: {outputs['pedal_position']:8.3f}")
        print(f"  Velocity Error: {outputs['velocity_error']:8.3f} m/s")
        print()
    
    # Show PID gains
    print(f"PID Gains - Kp: {driver.Kp}, Ki: {driver.Ki}, Kd: {driver.Kd}, N: {driver.N}")
    
    # Optionally plot NEDC cycle (if matplotlib is available)
    try:
        driver.plot_nedc_cycle()
    except:
        print("Could not plot NEDC cycle")
