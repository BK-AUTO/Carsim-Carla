from driver_model import DriverModel
from Longitudinal_model import LongitudinalModel
from front_tire_model import FrontTireModel
from rear_tire_model import RearTireModel
from braking_model import BrakingModel
from motor_model import MotorModel
from battery_model import BatteryModel
from power_distribution import calculate_traction_ratios
import numpy as np

def integrated_vehicle_simulation(vehicle_params=None, simulation_params=None, tire_params=None):
    """
    Integrated simulation combining Driver Model and Longitudinal Model
    
    Args:
        vehicle_params (dict): Vehicle parameters for LongitudinalModel
        simulation_params (dict): Simulation control parameters  
        tire_params (dict): Tire force parameters
    """
    
    # Default vehicle parameters (có thể thay đổi từ main)
    default_vehicle_params = {
        'vehicle_mass': 2530,               # kg - khối lượng xe
        'frontal_area': 0.7324,               # m² - diện tích cản gió  
        'drag_coefficient': 0.23,           # hệ số cản gió
        'air_density': 1.24,               # kg/m³ - mật độ không khí
        'rolling_resistance_coeff': 0.016,  # hệ số cản lăn
        'road_grade': 0,                    # radians - độ dốc đường
        'rotational_inertia_coeff': 1.05,   # hệ số quán tính quay
        'id': 0.5,                          # Gradability component base value
        'f': 0.5                            # Additional Gradability component resistance factor
    }
    
    # Front tire parameters
    default_front_tire_params = {
        'wheelbase_length': 2.95,          # m - chiều dài cơ sở
        'cg_height': 0.650,                 # m - chiều cao trọng tâm
        'tire_radius': 0.365,               # m - bán kính bánh xe
        'gear_ratio_tlc': 1.1,             # tỷ số truyền
        'gear_ratio_hs': 10.4180,               # tỷ số tốc độ cao
        'inertia_wheel': 1.5,               # kg*m² - mô men quán tính bánh xe
        'inertia_diff': 0.05,               # kg*m² - mô men quán tính vi sai
        'inertia_gearbox': 0.02,            # kg*m² - mô men quán tính hộp số
        'inertia_motor': 0.1                # kg*m² - mô men quán tính động cơ
    }
    
    # Rear tire parameters (same as front for now)
    default_rear_tire_params = {
        'wheelbase_length': 2.875,          # m - chiều dài cơ sở
        'cg_height': 0.540,                 # m - chiều cao trọng tâm
        'tire_radius': 0.334,               # m - bán kính bánh xe
        'gear_ratio_tlc': 9.73,             # tỷ số truyền
        'gear_ratio_hs': 1.0,               # tỷ số tốc độ cao
        'inertia_wheel': 1.9,               # kg*m² - mô men quán tính bánh xe
        'inertia_diff': 0.07,               # kg*m² - mô men quán tính vi sai
        'inertia_gearbox': 0.0196,            # kg*m² - mô men quán tính hộp số
        'inertia_motor': 0.01                # kg*m² - mô men quán tính động cơ
    }
    
    # Default simulation parameters  
    default_simulation_params = {
        'dt': 0.1,                          # seconds - bước thời gian
        'total_time': 100.0,                # seconds - thời gian mô phỏng
        'max_traction_force': 5000,         # N - lực kéo tối đa
        'max_braking_force': 8000           # N - lực phanh tối đa
    }
    
    # Default tire parameters
    default_tire_params = {
        'front_vertical_load': 8000.0,      # N - tải trọng trục trước
        'rear_vertical_load': 12000.0,      # N - tải trọng trục sau  
        'front_traction_ratio': 0.4,        # tỷ lệ lực kéo trục trước
        'rear_traction_ratio': 0.6,         # tỷ lệ lực kéo trục sau
        'front_braking_ratio': 0.6,         # tỷ lệ phanh trục trước
        'rear_braking_ratio': 0.4           # tỷ lệ phanh trục sau
    }
    
    # Merge with user parameters
    if vehicle_params:
        default_vehicle_params.update(vehicle_params)
    if simulation_params:
        default_simulation_params.update(simulation_params)  
    if tire_params:
        default_tire_params.update(tire_params)
    
    # Create model instances with parameters
    driver = DriverModel()
    longitudinal = LongitudinalModel(**default_vehicle_params)
    front_tire = FrontTireModel(
        vehicle_mass=default_vehicle_params['vehicle_mass'],
        wheelbase_length=default_front_tire_params['wheelbase_length'],
        cg_height=default_front_tire_params['cg_height'],
        tire_radius=default_front_tire_params['tire_radius'],
        gear_ratio_tlc=default_front_tire_params['gear_ratio_tlc'],
        gear_ratio_hs=default_front_tire_params['gear_ratio_hs'],
        inertia_wheel=default_front_tire_params['inertia_wheel'],
        inertia_diff=default_front_tire_params['inertia_diff'],
        inertia_gearbox=default_front_tire_params['inertia_gearbox'],
        inertia_motor=default_front_tire_params['inertia_motor']
    )
    rear_tire = RearTireModel(
        vehicle_mass=default_vehicle_params['vehicle_mass'],
        wheelbase_length=default_rear_tire_params['wheelbase_length'],
        cg_height=default_rear_tire_params['cg_height'],
        tire_radius=default_rear_tire_params['tire_radius'],
        gear_ratio_tlc=default_rear_tire_params['gear_ratio_tlc'],
        gear_ratio_hs=default_rear_tire_params['gear_ratio_hs'],
        inertia_wheel=default_rear_tire_params['inertia_wheel'],
        inertia_diff=default_rear_tire_params['inertia_diff'],
        inertia_gearbox=default_rear_tire_params['inertia_gearbox'],
        inertia_motor=default_rear_tire_params['inertia_motor']
    )
    braking = BrakingModel(
        vehicle_mass=default_vehicle_params['vehicle_mass'],
        wheelbase_length=default_front_tire_params['wheelbase_length'],
        cg_height=default_front_tire_params['cg_height'],
        tire_radius=default_front_tire_params['tire_radius'],
        gear_ratio_tlc=default_front_tire_params['gear_ratio_tlc'],
        gear_ratio_hs=default_front_tire_params['gear_ratio_hs'],
        inertia_wheel_front=default_front_tire_params['inertia_wheel'],
        inertia_wheel_rear=default_rear_tire_params['inertia_wheel']
    )
    motor = MotorModel(
        max_motor_torque=520.0,  # 520 Nm max torque
        omega_c=628.32           # 628.32 rad/s critical speed
    )
    
    battery = BatteryModel(
        initial_soc=0.95,  # 95% SOC as fraction
        capacity_ah=231.7,  # 231.7Ah capacity as per Simulink K factor
        initial_voltage=350.0
    )
    
    # Simulation setup
    dt = default_simulation_params['dt']
    total_time = default_simulation_params['total_time']
    time_steps = np.arange(0, total_time + dt, dt)
    
    # Initialize states
    current_velocity_ms = 0.0
    front_traction_force = 0.0
    rear_traction_force = 0.0
    
    print("Integrated Vehicle Simulation - All Outputs (with Front & Rear Tire Models + Braking + Battery)")
    print("=" * 260)
    print(f"{'Time':<6} {'DesVel':<8} {'ActVel':<8} {'Pedal':<8} {'Accel':<8} {'Dist':<8} {'Fzf':<8} {'Fxf':<8} {'Fzr':<8} {'Fxr':<8} {'SlipF':<8} {'SlipR':<8} {'BrMode':<10} {'RegenT':<10} {'MechT':<10} {'SOC':<8} {'BatV':<8} {'BatI':<8} {'BatP(W)':<10}")
    print(f"{'(s)':<6} {'(km/h)':<8} {'(km/h)':<8} {'pos':<8} {'(m/s²)':<8} {'(m)':<8} {'(N)':<8} {'(N)':<8} {'(N)':<8} {'(N)':<8} {'':<8} {'':<8} {'':<10} {'(Nm)':<10} {'(Nm)':<10} {'(%)':<8} {'(V)':<8} {'(A)':<8} {'(W)':<10}")
    print("-" * 270)
    
    # Storage for all outputs
    simulation_results = []
    
    # Initialize limited torques
    limited_front_torque = 0.0
    limited_rear_torque = 0.0
    
    # Main simulation loop
    for t in time_steps:
        # 1. Get driver outputs (desired velocity and pedal position)
        driver_outputs = driver.get_outputs(t, current_velocity_ms)
        
        # 2. Convert pedal position to motor torque and braking torque
        pedal_position = driver_outputs['pedal_position']
        
        # Get previous iteration data for braking model
        if len(simulation_results) > 0:
            prev_front_motor_velocity = simulation_results[-1]['front_tire_outputs']['front_motor_angular_velocity']
            prev_rear_motor_velocity = simulation_results[-1]['rear_tire_outputs']['rear_motor_angular_velocity']
            prev_front_vertical_load = simulation_results[-1]['front_tire_outputs']['front_vertical_load']
            prev_rear_vertical_load = simulation_results[-1]['rear_tire_outputs']['rear_vertical_load']
            prev_vehicle_acceleration = simulation_results[-1]['longitudinal_outputs']['vehicle_acceleration']
            
            # Use current state of tires for motor/battery calculations
            # Note: front_tire.front_tire_angular_velocity is the state variable
            current_front_motor_w = front_tire.front_tire_angular_velocity * front_tire.i_tlc * front_tire.i_hs
            current_rear_motor_w = rear_tire.rear_tire_angular_velocity * rear_tire.i_tlc * rear_tire.i_hs
        else:
            prev_front_motor_velocity = 0.0
            prev_rear_motor_velocity = 0.0
            prev_front_vertical_load = default_vehicle_params['vehicle_mass'] * 9.81 * 0.4
            prev_rear_vertical_load = default_vehicle_params['vehicle_mass'] * 9.81 * 0.6
            prev_vehicle_acceleration = 0.0
            current_front_motor_w = 0.0
            current_rear_motor_w = 0.0
        
        # Calculate motor torque and braking torque using BrakingModel
        if pedal_position > 0:
            # Acceleration - apply motor torque
            # Note: These are requested torques, will be limited by battery later
            front_motor_torque_req = pedal_position * 200.0
            rear_motor_torque_req = pedal_position * 200.0
            front_mechanical_braking_torque = 0.0
            rear_mechanical_braking_torque = 0.0
            # Create dummy braking outputs for consistency
            braking_outputs = {
                'front_regen_torque': 0.0,
                'front_mechanical_torque': 0.0,
                'rear_regen_torque': 0.0,
                'rear_mechanical_torque': 0.0,
                'braking_mode': 'none',
                'emergency_braking': False,
                'total_regen_torque': 0.0,
                'total_mechanical_torque': 0.0,
                'total_braking_torque': 0.0,
                'total_regen_power_kw': 0.0
            }
        else:
            # Braking - use BrakingModel
            front_motor_torque_req = 0.0
            rear_motor_torque_req = 0.0
            
            # Get braking torques from braking model
            braking_outputs = braking.get_outputs(
                pedal_position,
                current_velocity_ms,
                prev_front_motor_velocity,
                prev_rear_motor_velocity,
                prev_front_vertical_load,
                prev_rear_vertical_load,
                prev_vehicle_acceleration
            )
            
            # Use mechanical braking torques directly (without adding regen torque)
            front_mechanical_braking_torque = braking_outputs['front_mechanical_torque']
            rear_mechanical_braking_torque = braking_outputs['rear_mechanical_torque']
            
        # 3. Get motor model outputs (calculate power from torques)
        # Use current motor angular velocity (from tire state)
        motor_outputs = motor.get_outputs(
            current_front_motor_w,
            current_rear_motor_w,
            pedal_position,
            braking_outputs['front_regen_torque'],
            braking_outputs['rear_regen_torque']
        )
        
        # Calculate dynamic traction ratios based on required power
        front_traction_ratio, rear_traction_ratio = calculate_traction_ratios(
            motor_outputs['front_motor_required_power'],
            motor_outputs['rear_motor_required_power']
        )
        
        # 4. Update battery model
        # Calculate limited torque based on required power
        battery_outputs = battery.update_states(
            motor_outputs['front_motor_required_power'],
            motor_outputs['rear_motor_required_power'],
            motor_outputs['front_motor_required_torque'],
            motor_outputs['rear_motor_required_torque'],
            dt,
            w_front=current_front_motor_w,
            w_rear=current_rear_motor_w,
            velocity=current_velocity_ms,
            acceleration=prev_vehicle_acceleration,
            front_traction_ratio=front_traction_ratio,
            rear_traction_ratio=rear_traction_ratio
        )
        
        # Get limited torques from battery
        limited_front_torque = battery_outputs['limited_front_torque']
        limited_rear_torque = battery_outputs['limited_rear_torque']
        
        # 5. Get front tire model outputs (this calculates Fxf based on slip and Fzf)
        # Use LIMITED torque from battery
        front_tire_outputs = front_tire.get_outputs(
            vehicle_acceleration=prev_vehicle_acceleration,
            vehicle_velocity_ms=current_velocity_ms,
            front_motor_torque=limited_front_torque, # Use limited torque
            front_mechanical_braking_torque=front_mechanical_braking_torque,
            current_time=t,
            dt=dt
        )
        
        # 5b. Get rear tire model outputs (this calculates Fxr based on slip and Fzr)
        # Use LIMITED torque from battery
        rear_tire_outputs = rear_tire.get_outputs(
            vehicle_acceleration=prev_vehicle_acceleration,
            vehicle_velocity_ms=current_velocity_ms,
            rear_motor_torque=limited_rear_torque, # Use limited torque
            rear_mechanical_braking_torque=rear_mechanical_braking_torque,
            current_time=t,
            dt=dt
        )
        
        # 6. Get longitudinal model outputs (vehicle dynamics)
        # Use BOTH front_traction_force (Fxf) and front_vertical_load (Fzf) from tire model
        # AND rear_traction_force (Fxr) and rear_vertical_load (Fzr) from rear tire model
        front_traction_force = front_tire_outputs['front_traction_force']  # ← Fxf từ tire model!
        rear_traction_force = rear_tire_outputs['rear_traction_force']  # ← Fxr từ rear tire model!
        longitudinal_outputs = longitudinal.get_outputs(
            front_traction_force,  # ← Sử dụng Fxf từ front_tire_model
            rear_traction_force,   # ← Sử dụng Fxr từ rear_tire_model
            front_tire_outputs['front_vertical_load'],  # Use calculated Fzf from tire model
            rear_tire_outputs['rear_vertical_load'],    # ← Use calculated Fzr from rear tire model!
            t,
            id=default_vehicle_params['id'],
            f=default_vehicle_params['f']
        )
        
        # 7. Update current velocity for next iteration (FEEDBACK LOOP)
        current_velocity_ms = longitudinal_outputs['vehicle_velocity_ms']
        current_velocity_kmh = longitudinal_outputs['vehicle_velocity_kmh']
        
        # 8. Store all outputs
        combined_outputs = {
            'time': t,
            'driver_outputs': driver_outputs,
            'longitudinal_outputs': longitudinal_outputs,
            'front_tire_outputs': front_tire_outputs,
            'rear_tire_outputs': rear_tire_outputs,
            'braking_outputs': braking_outputs,
            'motor_outputs': motor_outputs,
            'battery_outputs': battery_outputs,
            'front_traction_force': front_traction_force,
            'rear_traction_force': rear_traction_force
        }
        simulation_results.append(combined_outputs)
        
        # 9. Print results every 1 second
        if abs(t % 1.0) < dt/2:  # Print every 1 second
            print(f"{t:<6.1f} {driver_outputs['desired_velocity_kmh']:<8.1f} "
                  f"{current_velocity_kmh:<8.1f} {pedal_position:<8.3f} "
                  f"{longitudinal_outputs['vehicle_acceleration']:<8.4f} "
                  f"{longitudinal_outputs['distance_traveled']:<8.1f} "
                  f"{front_tire_outputs['front_vertical_load']:<8.1f} "
                  f"{front_tire_outputs['front_traction_force']:<8.1f} "
                  f"{rear_tire_outputs['rear_vertical_load']:<8.1f} "
                  f"{rear_tire_outputs['rear_traction_force']:<8.1f} "
                  f"{front_tire_outputs['front_tire_slip']:<8.4f} "
                  f"{rear_tire_outputs['rear_tire_slip']:<8.4f} "
                  f"{braking_outputs['braking_mode']:<10} "
                  f"{braking_outputs['total_regen_torque']:<10.1f} "
                  f"{braking_outputs['total_mechanical_torque']:<10.1f} "
                  f"{battery_outputs['soc']*100:<8.1f} "
                  f"{battery_outputs['voltage']:<8.1f} "
                  f"{battery_outputs['current']:<8.1f} "
                  f"{battery_outputs['power_watts']:<10.1f}")
    
    print("-" * 260)
    print("Simulation completed!")
    print(f"Final velocity: {current_velocity_kmh:.1f} km/h")
    print(f"Total distance: {longitudinal_outputs['distance_traveled']:.1f} m")
    print(f"Final SOC: {battery_outputs['soc']*100:.2f}%")
    print(f"Total Energy Consumed: {battery_outputs['total_energy_consumption_kwh']:.4f} kWh")
    print(f"Total Regen Energy: {battery_outputs['total_regen_energy_kwh']:.4f} kWh")
    print(f"Final front tire angular velocity: {front_tire_outputs['front_tire_angular_velocity']:.2f} rad/s")
    print(f"Final front motor angular velocity: {front_tire_outputs['front_motor_angular_velocity']:.2f} rad/s")
    print(f"Final rear tire angular velocity: {rear_tire_outputs['rear_tire_angular_velocity']:.2f} rad/s")
    print(f"Final rear motor angular velocity: {rear_tire_outputs['rear_motor_angular_velocity']:.2f} rad/s")
    print(f"Final front motor required torque: {motor_outputs['front_motor_required_torque']:.2f} Nm")
    print(f"Final rear motor required torque: {motor_outputs['rear_motor_required_torque']:.2f} Nm")
    print(f"Final front motor required power: {motor_outputs['front_motor_required_power']:.2f} kW")
    print(f"Final rear motor required power: {motor_outputs['rear_motor_required_power']:.2f} kW")
    print(f"Final total motor power: {motor_outputs['total_motor_power']:.2f} kW")
    print(f"PID Gains - Kp: {driver.Kp}, Ki: {driver.Ki}, Kd: {driver.Kd}, N: {driver.N}")
    
    # Return all simulation results for further analysis
    return simulation_results

def print_detailed_outputs(simulation_results, time_index):
    """
    Print detailed outputs from all models at a specific time
    """
    result = simulation_results[time_index]
    t = result['time']
    driver_out = result['driver_outputs']
    long_out = result['longitudinal_outputs']
    front_tire_out = result['front_tire_outputs']
    rear_tire_out = result['rear_tire_outputs']
    
    print(f"\n=== DETAILED OUTPUTS AT t={t:.1f}s ===")
    print(f"\nDRIVER MODEL OUTPUTS:")
    print(f"  - Desired Velocity: {driver_out['desired_velocity_kmh']:.2f} km/h ({driver_out['desired_velocity_ms']:.2f} m/s)")
    print(f"  - Desired Acceleration: {driver_out['desired_acceleration']:.4f} m/s²")
    print(f"  - Desired Distance: {driver_out['desired_distance']:.1f} m")
    print(f"  - Pedal Position: {driver_out['pedal_position']:.4f}")
    print(f"  - Velocity Error: {driver_out['velocity_error']:.3f} m/s")
    
    print(f"\nLONGITUDINAL MODEL OUTPUTS:")
    print(f"  - Vehicle Acceleration: {long_out['vehicle_acceleration']:.4f} m/s²")
    print(f"  - Vehicle Velocity: {long_out['vehicle_velocity_kmh']:.2f} km/h ({long_out['vehicle_velocity_ms']:.2f} m/s)")
    print(f"  - Distance Traveled: {long_out['distance_traveled']:.1f} m")
    print(f"  - Total Traction Force: {long_out['total_traction_force']:.1f} N")
    print(f"  - Total Resistance Force: {long_out['total_resistance_force']:.1f} N")
    print(f"  - Net Force: {long_out['net_force']:.1f} N")
    print(f"  - Aerodynamic Drag: {long_out['aerodynamic_drag']:.1f} N")
    print(f"  - Rolling Resistance: {long_out['rolling_resistance']:.1f} N")
    print(f"  - Grade Resistance: {long_out['grade_resistance']:.1f} N")
    
    print(f"\nFRONT TIRE MODEL OUTPUTS:")
    print(f"  - Front Vertical Load (Fzf): {front_tire_out['front_vertical_load']:.1f} N")
    print(f"  - Front Tire Slip: {front_tire_out['front_tire_slip']:.4f}")
    print(f"  - Front Traction Force (Fxf): {front_tire_out['front_traction_force']:.1f} N")
    print(f"  - Front Tire Angular Velocity: {front_tire_out['front_tire_angular_velocity']:.2f} rad/s")
    print(f"  - Front Tire Angular Acceleration: {front_tire_out['front_tire_angular_acceleration']:.2f} rad/s²")
    print(f"  - Front Motor Angular Velocity: {front_tire_out['front_motor_angular_velocity']:.2f} rad/s")
    
    print(f"\nREAR TIRE MODEL OUTPUTS:")
    print(f"  - Rear Vertical Load (Fzr): {rear_tire_out['rear_vertical_load']:.1f} N")
    print(f"  - Rear Tire Slip: {rear_tire_out['rear_tire_slip']:.4f}")
    print(f"  - Rear Traction Force (Fxr): {rear_tire_out['rear_traction_force']:.1f} N")
    print(f"  - Rear Tire Angular Velocity: {rear_tire_out['rear_tire_angular_velocity']:.2f} rad/s")
    print(f"  - Rear Tire Angular Acceleration: {rear_tire_out['rear_tire_angular_acceleration']:.2f} rad/s²")
    print(f"  - Rear Motor Angular Velocity: {rear_tire_out['rear_motor_angular_velocity']:.2f} rad/s")
    
    print(f"\nBRAKING MODEL OUTPUTS:")
    braking_out = result['braking_outputs']
    print(f"  - Braking Mode: {braking_out['braking_mode']}")
    print(f"  - Emergency Braking: {braking_out['emergency_braking']}")
    print(f"  - Front Regen Torque: {braking_out['front_regen_torque']:.2f} Nm")
    print(f"  - Front Mechanical Torque: {braking_out['front_mechanical_torque']:.2f} Nm")
    print(f"  - Rear Regen Torque: {braking_out['rear_regen_torque']:.2f} Nm")
    print(f"  - Rear Mechanical Torque: {braking_out['rear_mechanical_torque']:.2f} Nm")
    print(f"  - Total Regen Torque: {braking_out['total_regen_torque']:.2f} Nm")
    print(f"  - Total Mechanical Torque: {braking_out['total_mechanical_torque']:.2f} Nm")
    print(f"  - Total Braking Torque: {braking_out['total_braking_torque']:.2f} Nm")
    print(f"  - Regen Power: {braking_out['total_regen_power_kw']:.2f} kW")
    
    print(f"\nMOTOR MODEL OUTPUTS:")
    motor_out = result['motor_outputs']
    print(f"  - Front Motor Required Torque: {motor_out['front_motor_required_torque']:.2f} Nm")
    print(f"  - Rear Motor Required Torque: {motor_out['rear_motor_required_torque']:.2f} Nm")
    print(f"  - Front Motor Required Power: {motor_out['front_motor_required_power']:.2f} kW")
    print(f"  - Rear Motor Required Power: {motor_out['rear_motor_required_power']:.2f} kW")
    print(f"  - Total Motor Power: {motor_out['total_motor_power']:.2f} kW")
    
    print(f"\nBATTERY MODEL OUTPUTS:")
    bat_out = result['battery_outputs']
    print(f"  - SOC: {bat_out['soc']*100:.2f}%")
    print(f"  - Voltage: {bat_out['voltage']:.2f} V")
    print(f"  - Current: {bat_out['current']:.2f} A")
    print(f"  - OCV: {bat_out['battery_voltage_ocv']:.2f} V")
    print(f"  - Internal Resistance: {bat_out['internal_resistance']:.4f} Ohm")
    print(f"  - Total Energy Consumed: {bat_out['total_energy_consumption_kwh']:.4f} kWh")
    print(f"  - Total Regen Energy: {bat_out['total_regen_energy_kwh']:.4f} kWh")
    print(f"  - Power Loss: {bat_out['motor_power_loss_kwh']:.4f} kWh")
    
    print(f"\nFORCE DISTRIBUTION:")
    print(f"  - Front Traction Force: {result['front_traction_force']:.1f} N")
    print(f"  - Rear Traction Force: {result['rear_traction_force']:.1f} N")

# Example usage and testing
if __name__ == "__main__":
    
    # CÁC THAM SỐ CÓ THỂ ĐIỀU CHỈNH TẠI ĐÂY:
    
    # Tham số xe (Vehicle Parameters)
    custom_vehicle_params = {
        'vehicle_mass': 2530,               # kg - có thể thay đổi khối lượng xe
        'frontal_area': 2.58,               # m² - diện tích cản gió
        'drag_coefficient': 0.23,           # hệ số cản gió (0.2-0.4)
        'rolling_resistance_coeff': 0.016,  # hệ số cản lăn (0.01-0.02)
        'road_grade': 0,                    # độ dốc (0 = đường bằng)
    }
    
    # Tham số mô phỏng (Simulation Parameters)  
    custom_simulation_params = {
        'dt': 0.1,                          # bước thời gian (giây)
        'total_time': 100.0,                # thời gian mô phỏng (giây)
        'max_traction_force': 5000,         # lực kéo tối đa (N)
        'max_braking_force': 8000           # lực phanh tối đa (N)
    }
    
    # Tham số lốp (Tire Parameters)
    custom_tire_params = {
        'front_vertical_load': 8000.0,      # tải trọng trục trước (N)
        'rear_vertical_load': 12000.0,      # tải trọng trục sau (N)
        'front_traction_ratio': 0.4,        # 40% lực kéo ở trước
        'rear_traction_ratio': 0.6,         # 60% lực kéo ở sau
        'front_braking_ratio': 0.6,         # 60% lực phanh ở trước  
        'rear_braking_ratio': 0.4           # 40% lực phanh ở sau
    }
    
    print("🚗 CHẠY MÔ PHỎNG VỚI THAM SỐ TÙY CHỈNH")
    print(f"Khối lượng xe: {custom_vehicle_params['vehicle_mass']} kg")
    print(f"Hệ số cản gió: {custom_vehicle_params['drag_coefficient']}")
    print(f"Hệ số cản lăn: {custom_vehicle_params['rolling_resistance_coeff']}")
    print(f"Thời gian mô phỏng: {custom_simulation_params['total_time']} giây")
    print("-" * 60)
    
    # Run integrated simulation với tham số tùy chỉnh
    results = integrated_vehicle_simulation(
        vehicle_params=custom_vehicle_params,
        simulation_params=custom_simulation_params, 
        tire_params=custom_tire_params
    )
    
    print("\n" + "="*80)
    print("DETAILED OUTPUT ANALYSIS")
    print("="*80)
    
    # Show detailed outputs at key time points
    key_times = [150, 500, 940]  # Different phases of NEDC cycle
    for time_idx in key_times:
        if time_idx < len(results):
            print_detailed_outputs(results, time_idx)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate some summary statistics
    max_velocity = max([r['longitudinal_outputs']['vehicle_velocity_kmh'] for r in results])
    max_acceleration = max([r['longitudinal_outputs']['vehicle_acceleration'] for r in results])
    min_acceleration = min([r['longitudinal_outputs']['vehicle_acceleration'] for r in results])
    final_distance = results[-1]['longitudinal_outputs']['distance_traveled']
    
    print(f"Maximum Velocity: {max_velocity:.1f} km/h")
    print(f"Maximum Acceleration: {max_acceleration:.3f} m/s²")
    print(f"Maximum Deceleration: {min_acceleration:.3f} m/s²")
    print(f"Total Distance: {final_distance:.1f} m")
    
    # Test individual models
    driver = DriverModel()
    print(f"\nPID Controller Settings: Kp={driver.Kp}, Ki={driver.Ki}, Kd={driver.Kd}, N={driver.N}")