import numpy as np

# Given data
normalized_POG_errors = [
    0.1012, 0.1366, 0.0970, 0.2183, 0.0954, 0.1093, 0.1048,
    0.1229, 0.1187, 0.1247, 0.0984, 0.1168, 0.1242, 0.1099, 0.1630
]

screen_width = 1280
par_a_w = 0.2238
par_b_w = -3.2354
par_a_h = 0.2238
par_b_h = 0

def calculate_real_world_error(normalized_POG_error, screen_width, par_a_w, par_b_w, par_a_h, par_b_h):
    # Step 1: Convert Normalized Error to Pixel Error
    pixel_error = normalized_POG_error * screen_width

    # Step 2: Convert Pixel Error to Real-World Coordinates
    real_x_error = (pixel_error * par_a_w) + par_b_w
    real_y_error = (pixel_error * par_a_h) + par_b_h

    # Step 3: Calculate Euclidean Distance
    euclidean_distance = np.sqrt(real_x_error**2 + real_y_error**2)
    return euclidean_distance

# Calculate the real-world error for each subject
real_world_errors = [
    calculate_real_world_error(error, screen_width, par_a_w, par_b_w, par_a_h, par_b_h)
    for error in normalized_POG_errors
]

# Calculate the average error over all subjects
average_error = np.mean(real_world_errors)

# Print the results
for i, (norm_error, real_error) in enumerate(zip(normalized_POG_errors, real_world_errors)):
    print(f"Subject {i+1} - Normalized POG Error: {norm_error}, Real-World Error: {real_error:.2f} mm")

print(f"\nAverage Real-World Error over all subjects: {average_error:.2f} mm")
