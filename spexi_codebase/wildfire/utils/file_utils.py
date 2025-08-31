
import os
import mmcv
import shutil

def create_output_dirs(*dirs):
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def get_relative_path(root, base):
    return os.path.relpath(root, base)



def save_color_mask(color_mask, output_folder, relative_path, file_name):
    # Save the color mask to the specified folder, maintaining the subfolder structure.

    # Create the subfolder path inside the output directory
    output_subfolder = os.path.join(output_folder, relative_path)

    # Ensure the subfolder exists
    os.makedirs(output_subfolder, exist_ok=True)

    # Construct the full file path for the color mask
    output_mask_path = os.path.join(output_subfolder, file_name)

    # Save the color mask to the specified file path
    mmcv.imwrite(color_mask, output_mask_path)

    return output_mask_path


def update_index_js(input_folder):
    """
    Updates the `index.js` file to set the `urlPrefix` dynamically based on the `input_folder`.

    Parameters:
        input_folder (str): The value to set as the `urlPrefix`.
    """
    js_file_path = 'output/index.js'  # Replace with the actual path to your `index.js` file

    try:
        # Read the JavaScript file
        with open(js_file_path, 'r') as file:
            js_content = file.readlines()
        
        # Update the `urlPrefix` line
        updated_content = []
        for line in js_content:
            if 'var urlPrefix =' in line:
                updated_content.append(f'var urlPrefix = "{input_folder}";\n')  # Replace with the new value
            else:
                updated_content.append(line)

        # Write the updated content back to the file
        with open(js_file_path, 'w') as file:
            file.writelines(updated_content)

        print(f"Successfully updated {js_file_path} with urlPrefix = {input_folder}")

    except Exception as e:
        print(f"Failed to update {js_file_path}: {e}")
        
        


