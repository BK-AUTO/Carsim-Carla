def calculate_traction_ratios(front_required_power, rear_required_power):
    """
    Calculate front and rear traction ratios based on required power.
    
    Args:
        front_required_power (float): Front motor required power (kW)
        rear_required_power (float): Rear motor required power (kW)
        
    Returns:
        tuple: (front_ratio, rear_ratio) saturated between 0 and 1
    """
    f = front_required_power
    r = rear_required_power
    
    total = f + r
    
    # Avoid division by zero
    if abs(total) < 1e-9:
        # If total power is 0, assume equal distribution or maintain previous state
        # Returning 0.5, 0.5 as a safe default
        return 0.5, 0.5
        
    r_f = f / total
    r_r = r / total
    
    # Saturation 0 to 1
    r_f = max(0.0, min(1.0, r_f))
    r_r = max(0.0, min(1.0, r_r))
    
    return r_f, r_r
