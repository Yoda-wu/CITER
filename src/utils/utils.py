def move_past_key_values_to_device(past_key_values, device_map):
    new_past_key_values = []
    for i, layer_past in enumerate(past_key_values):
        device = device_map.get(i, "cpu")
        new_past_key_values.append(tuple([p.to(device) for p in layer_past]))
    return new_past_key_values
