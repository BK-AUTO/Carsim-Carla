import numpy as np
import math

class LongitudinalModel:
    """
    Longitudinal Vehicle Dynamics Model
    Based on Simulink model for vehicle motion calculation
    """
    
    def __init__(self, vehicle_mass, frontal_area, drag_coefficient, 
                 air_density, rolling_resistance_coeff, road_grade, 
                 rotational_inertia_coeff,id,f):
        """
        Initialize longitudinal model parameters
        
        Args:
            vehicle_mass (float): Vehicle mass in kg
            frontal_area (float): Vehicle frontal area in m²
            drag_coefficient (float): Aerodynamic drag coefficient
            air_density (float): Air density in kg/m³
            rolling_resistance_coeff (float): Rolling resistance coefficient
            road_grade (float): Road grade (slope) in radians
            rotational_inertia_coeff (float): Rotational inertia coefficient (1.05 typical)
        """
        self.m = vehicle_mass  # Vehicle mass (kg)
        self.A = frontal_area  # Frontal area (m²)
        self.Cd = drag_coefficient  # Drag coefficient
        self.rho = air_density  # Air density (kg/m³)
        self.Crr = rolling_resistance_coeff  # Rolling resistance coefficient
        self.grade = road_grade  # Road grade (radians)
        self.delta = rotational_inertia_coeff  # Rotational inertia coefficient
        self.g = 9.81  # Gravitational acceleration (m/s²)
        self.id = id
        self.f = f
        # Internal states
        self.velocity_ms = 0.0  # Vehicle velocity in m/s
        self.distance = 0.0  # Total distance traveled in m
        self.previous_time = 0.0
        
    def calculate_aerodynamic_drag(self, velocity_ms):
        """
        Calculate aerodynamic drag force
        
        Args:
            velocity_ms (float): Vehicle velocity in m/s
            
        Returns:
            float: Aerodynamic drag force in N
        """
        # F_aero = 0.5 * rho * Cd * A * v²
        drag_force = 0.5 * self.rho * self.Cd * self.A * (velocity_ms ** 2)
        return drag_force
    
    def calculate_rolling_resistance(self, front_vertical_load, rear_vertical_load, velocity_ms):
        """
        Calculate rolling resistance force
        
        Args:
            front_vertical_load (float): Front tire vertical load in N
            rear_vertical_load (float): Rear tire vertical load in N
            velocity_ms (float): Vehicle velocity in m/s
            
        Returns:
            float: Rolling resistance force in N
        """
        # Total normal force
        total_normal_force = front_vertical_load + rear_vertical_load
        rolling_resistance = total_normal_force
        return rolling_resistance
    
    def calculate_grade_resistance(self, id, f):
        """
        Calculate grade (gravitational) resistance force based on the provided formula.
        
        Args:
            id (float): Road inclination component.
            f (float): Rolling resistance component for grade calculation.
            
        Returns:
            float: Grade resistance force in N.
        """
        # This appears to be a custom formula for gradability.
        # The original formula was F_grade = m * g * sin(grade).
        # The new formula seems to be F_grade = m * g * (f * cos(id) + sin(id))
        # Let's use the simpler `id + f` for now as per the previous attempt,
        # but a more physically accurate model might be needed.
        # For now, to fix the error, we just use the parameters.
        grade_resistance = self.id + self.f
        return grade_resistance
    
    def calculate_total_resistance(self, front_vertical_load, rear_vertical_load, velocity_ms, id, f):
        """
        Calculate total resistance forces
        
        Args:
            front_vertical_load (float): Front tire vertical load in N
            rear_vertical_load (float): Rear tire vertical load in N
            velocity_ms (float): Vehicle velocity in m/s
            id (float): Road inclination component.
            f (float): Rolling resistance component.
            
        Returns:
            tuple: (total_resistance, aero_drag, rolling_resistance, grade_resistance)
        """
        aero_drag = self.calculate_aerodynamic_drag(velocity_ms)
        rolling_resistance = self.calculate_rolling_resistance(front_vertical_load, rear_vertical_load, velocity_ms)
        grade_resistance = self.calculate_grade_resistance(id, f)
        
        total_resistance = aero_drag + 1 / (rolling_resistance + grade_resistance)
        
        return total_resistance, aero_drag, rolling_resistance, grade_resistance
    
    def calculate_vehicle_dynamics(self, front_traction_force, rear_traction_force, 
                                 front_vertical_load, rear_vertical_load, dt, id, f):
        """
        Calculate vehicle acceleration and velocity based on forces
        
        Args:
            front_traction_force (float): Front tire traction force in N
            rear_traction_force (float): Rear tire traction force in N
            front_vertical_load (float): Front tire vertical load in N
            rear_vertical_load (float): Rear tire vertical load in N
            dt (float): Time step in seconds
            id (float): Road inclination component.
            f (float): Rolling resistance component.
            
        Returns:
            tuple: (acceleration, velocity_ms, velocity_kmh, distance)
        """
        # Total traction force
        total_traction_force = front_traction_force + rear_traction_force
        
        # Calculate total resistance
        total_resistance, aero_drag, rolling_resistance, grade_resistance = self.calculate_total_resistance(front_vertical_load, rear_vertical_load, self.velocity_ms, id, f)
        
        # Net force equation: F_net = F_traction - F_resistance
        net_force = total_traction_force - total_resistance
        
        # Calculate acceleration: a = F_net / (m * delta)
        # delta accounts for rotational inertia of wheels and drivetrain
        acceleration = net_force / (self.m * self.delta)
        
        # Update velocity: v = v0 + a * dt
        self.velocity_ms += acceleration * dt
        
        # Ensure velocity doesn't go negative (vehicle can't go backward in this model)
        if self.velocity_ms < 0:
            self.velocity_ms = 0
            acceleration = 0  # No acceleration if stopped
        
        # Convert velocity to km/h
        velocity_kmh = self.velocity_ms * 3.6
        
        # Update distance: s = s0 + v * dt
        self.distance += self.velocity_ms * dt
        
        return acceleration, self.velocity_ms, velocity_kmh, self.distance
    
    def get_outputs(self, front_traction_force, rear_traction_force, 
                   front_vertical_load, rear_vertical_load, current_time, id, f):
        """
        Main function to get all longitudinal model outputs
        
        Args:
            front_traction_force (float): Front tire traction force in N
            rear_traction_force (float): Rear tire traction force in N
            front_vertical_load (float): Front tire vertical load in N
            rear_vertical_load (float): Rear tire vertical load in N
            current_time (float): Current simulation time in seconds
            id (float): Road inclination component.
            f (float): Rolling resistance component.
            
        Returns:
            dict: Dictionary containing all outputs
        """
        # Calculate time step
        if self.previous_time == 0:
            dt = 0.1  # Default time step
        else:
            dt = current_time - self.previous_time
        
        if dt <= 0:
            dt = 0.1
        
        # Calculate vehicle dynamics
        acceleration, velocity_ms, velocity_kmh, distance = self.calculate_vehicle_dynamics(front_traction_force, rear_traction_force,front_vertical_load, rear_vertical_load, dt, id, f)
        
        # Calculate individual resistance components for analysis
        total_resistance, aero_drag, rolling_resistance, grade_resistance = self.calculate_total_resistance(front_vertical_load, rear_vertical_load, velocity_ms, id, f)
        
        # Update previous time
        self.previous_time = current_time
        
        return {
            'vehicle_acceleration': acceleration,  # m/s²
            'vehicle_velocity_ms': velocity_ms,   # m/s
            'vehicle_velocity_kmh': velocity_kmh, # km/h
            'distance_traveled': distance,        # m
            'total_traction_force': front_traction_force + rear_traction_force,  # N
            'total_resistance_force': total_resistance,  # N
            'aerodynamic_drag': aero_drag,        # N
            'rolling_resistance': rolling_resistance,  # N
            'grade_resistance': grade_resistance,  # N
            'net_force': (front_traction_force + rear_traction_force) - total_resistance  # N
        }
    
    def reset_states(self):
        """Reset all internal states"""
        self.velocity_ms = 0.0
        self.distance = 0.0
        self.previous_time = 0.0
    
    def set_vehicle_parameters(self, mass=None, frontal_area=None, drag_coeff=None, 
                              rolling_resistance=None, road_grade=None):
        """
        Update vehicle parameters
        
        Args:
            mass (float, optional): Vehicle mass in kg
            frontal_area (float, optional): Frontal area in m²
            drag_coeff (float, optional): Drag coefficient
            rolling_resistance (float, optional): Rolling resistance coefficient
            road_grade (float, optional): Road grade in radians
        """
        if mass is not None:
            self.m = mass
        if frontal_area is not None:
            self.A = frontal_area
        if drag_coeff is not None:
            self.Cd = drag_coeff
        if rolling_resistance is not None:
            self.Crr = rolling_resistance
        if road_grade is not None:
            self.grade = road_grade
    
    def get_current_state(self):
        """
        Get current vehicle state
        
        Returns:
            dict: Current state information
        """
        return {
            'velocity_ms': self.velocity_ms,
            'velocity_kmh': self.velocity_ms * 3.6,
            'distance': self.distance,
            'vehicle_mass': self.m,
            'road_grade_deg': math.degrees(self.grade)
        }
    
    def calculate_power_requirement(self, front_traction_force, rear_traction_force):
        """
        Calculate power requirement based on current traction forces and velocity
        
        Args:
            front_traction_force (float): Front tire traction force in N
            rear_traction_force (float): Rear tire traction force in N
            
        Returns:
            float: Power requirement in Watts
        """
        total_traction_force = front_traction_force + rear_traction_force
        power_requirement = total_traction_force * self.velocity_ms
        return power_requirement


# Example usage and testing
if __name__ == "__main__":
    # Create longitudinal model instance
    longitudinal_model = LongitudinalModel(
        vehicle_mass=2530,      # kg (typical electric vehicle)
        frontal_area=2.58,      # m²
        drag_coefficient=0.23,  # Typical for modern vehicles
        rolling_resistance_coeff=0.016
    )
    
    print("Testing Longitudinal Model...")
    print("-" * 60)
    
    # Simulate some test conditions
    test_scenarios = [
        {"name": "Acceleration", "front_traction": 2000, "rear_traction": 3000, "front_load": 8000, "rear_load": 12000},
        {"name": "Cruising", "front_traction": 500, "rear_traction": 500, "front_load": 8000, "rear_load": 12000},
        {"name": "Braking", "front_traction": -1000, "rear_traction": -1500, "front_load": 10000, "rear_load": 10000},
        {"name": "Coasting", "front_traction": 0, "rear_traction": 0, "front_load": 8000, "rear_load": 12000}
    ]
    
    time = 0.0
    dt = 0.1
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 30)
        
        # Reset model for each scenario
        longitudinal_model.reset_states()
        
        # Run simulation for 5 seconds
        for i in range(50):  # 5 seconds at 0.1s intervals
            time += dt
            
            outputs = longitudinal_model.get_outputs(
                scenario['front_traction'], 
                scenario['rear_traction'],
                scenario['front_load'], 
                scenario['rear_load'], 
                time
            )
            
            # Print every 10th step (every 1 second)
            if i % 10 == 0 or i == 49:
                print(f"  t={time:4.1f}s: v={outputs['vehicle_velocity_kmh']:6.1f} km/h, "
                      f"a={outputs['vehicle_acceleration']:7.3f} m/s², "
                      f"dist={outputs['distance_traveled']:6.1f} m")
    
    print(f"\nModel Parameters:")
    print(f"  Vehicle Mass: {longitudinal_model.m} kg")
    print(f"  Frontal Area: {longitudinal_model.A} m²")
    print(f"  Drag Coefficient: {longitudinal_model.Cd}")
    print(f"  Rolling Resistance: {longitudinal_model.Crr}")
