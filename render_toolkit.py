import os
import re
import torch
import argparse
import mitsuba as mi

# read the variable GPU_RENDER from the environment, if it is set to 1, use GPU, otherwise use CPU
if os.environ.get('GPU_RENDER') == '1':
    print("Rendering with GPU")
    mi.set_variant('cuda_ad_rgb')
else:
    print("Rendering with CPU")
    mi.set_variant('scalar_rgb') # use CPU in default, as the GPU may be not available, or applied to other tasks
from mitsuba import ScalarTransform4f as T

# ==========================================
# Global Variables & XML Templates
# ==========================================
global_color_list = [
    [0.5488135, 0.71518937, 0.60276338],
    [0.43758721, 0.891773, 0.96366276],
    [0.56804456, 0.92559664, 0.07103606],
    [0.0871293, 0.0202184, 0.83261985],
    [0.77815675, 0.87001215, 0.57861834],
    [0.79915856, 0.46147936, 0.78052918],
    [0.11827443, 0.63992102, 0.14335329],
    [0.94466892, 0.52184832, 0.41466194],
    [0.26455561, 0.77423369, 0.45615033],
    [0.56843395, 0.0187898, 0.6176355],
    [0.6818203, 0.3595079, 0.43703195],
    [0.67063787, 0.21038256, 0.1289263],
    [0.31542835, 0.36371077, 0.57019677],
    [0.13818295, 0.19658236, 0.36872517],
    [0.82099323, 0.09710128, 0.83794491],
    [0.09394051, 0.5759465, 0.9292962],
    [0.0191932, 0.30157482, 0.66017354],
    [0.36756187, 0.43586493, 0.89192336],
    [0.80619399, 0.70388858, 0.10022689],
    [0.85580334, 0.01171408, 0.35997806]
]

xml_head = \
"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> </bsdf>
    
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

# ==========================================
# Rendering Functions
# ==========================================
def mitsuba_render(data_idx, gen_idx, save_dir, 
                   img_type="png",
                   sensor_width = 512,
                   sensor_height = 512,
                   sensor_sep = 25,            
                   phi = 225,  
                   radius = 2.5,
                   theta = -50,
                   spp = 512):
    # Load a scene
    scene = mi.load_file("{}/render_{}_{}.xml".format(save_dir, data_idx, gen_idx))

    def load_sensor(r, phi, theta):
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=origin,
                target=[0, 0, 0],
                up=[0, 0, 1]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': sensor_width,
                'height': sensor_height,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })
    sensor = load_sensor(radius, phi, theta)
    image = mi.render(scene, spp=spp, sensor=sensor)
    # Write the rendered image to an EXR/PNG file
    mi.util.write_bitmap("{}/render_{}_{}_{}.{}".format(save_dir, data_idx, gen_idx, phi, img_type), image)

# ==========================================
# Mesh Processing Functions
# ==========================================
def load_one_mesh(obj_path, color):
    mesh_str = \
    f"""
    <shape type="obj">
        <string name="filename" value="{obj_path}"/>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{color[0]},{color[1]},{color[2]}"/>
        </bsdf>
    </shape>
    """
    return mesh_str

def load_mesh_list(obj_list, color_list):
    mesh_str = ""
    for obj_path, color in zip(obj_list, color_list):
        mesh_str += load_one_mesh(obj_path, color)
    return mesh_str

def save_render_xml(obj_path_list, color_list, save_path):
    obj_load = load_mesh_list(obj_path_list, color_list)
    with open(save_path, "w") as f:
        f.write(xml_head)
        f.write(obj_load)
        f.write(xml_tail)

def get_color_list(num):
    # to keep consistency, we sequentially select color from global_color_list
    color_list = []
    for i in range(num):
        color_list.append(global_color_list[i % len(global_color_list)]) # prevent out of index
    return color_list

def lstrip_except_zero(s):
    # if a string only have zeros, then use lstrip will nothing left, 
    # to prevent this happen, we need to check if the string is all zeros
    if s.strip('0') == '':
        return '0'
    else:
        return s.lstrip('0')

def get_mesh(mesh_code):
    mesh_code_str = str(mesh_code)
    # check the integrity of the mesh_code_str
    assert len(mesh_code_str) == 16
    if mesh_code_str[0] != '1':
        raise ValueError(f'{mesh_code_str} is not a valid mesh code')
    # get related info
    category = mesh_code_str[1:3]
    obj_id = mesh_code_str[3:10]
    fracture_mode = 'fractured' if mesh_code_str[10:12] == '00' else 'mode'
    fracture_pattern = lstrip_except_zero(mesh_code_str[12:14])
    fracture_id = 'piece_' + lstrip_except_zero(mesh_code_str[14:16]) + '.obj'

    # create category dir dict
    category_dir_dict = {
        '01': 'animal_sculpture',
        '02': 'bowl',
        '03': 'coin',
        '04': 'human_sculpture',
        '05': 'jade',
        '06': 'jar',
        '07': 'musical_instrument',
        '08': 'plate',
        '09': 'stele',
        '10': 'tool',
        '11': 'vase',
        '12': 'weapon'
    }

    if category == '00':
        mesh_path = os.path.join('artifact', lstrip_except_zero(obj_id) + '_sf', fracture_mode + '_' + fracture_pattern, fracture_id)
    else:
        mesh_path = os.path.join(category + '-' + category_dir_dict[category], category + '-' + obj_id[-6:], fracture_mode + '_' + fracture_pattern, fracture_id)

    return mesh_path

def decode_mesh_code_list(mesh_code_list, mesh_dir=""):
    mesh_path_list = []
    mesh_path_wo_file_name_list = []
    for mesh_code in mesh_code_list:
        mesh_path = get_mesh(mesh_code.item())
        if mesh_dir != "":
            mesh_path = os.path.join(mesh_dir, mesh_path)
        mesh_path_list.append(mesh_path)
        mesh_path_wo_file_name = os.path.dirname(mesh_path)
        if mesh_path_wo_file_name not in mesh_path_wo_file_name_list:
            mesh_path_wo_file_name_list.append(mesh_path_wo_file_name)
    return mesh_path_list, mesh_path_wo_file_name_list

def get_set_info(test_num, main_part_set_path, cate):
    part_set_path = os.path.join(main_part_set_path, "{}_data_{}.pt".format(cate, test_num))
    part_dict = torch.load(part_set_path, map_location="cpu")
    part_number = part_dict["all_parts"].size(0)
    color_list = get_color_list(part_number)
    return color_list, part_dict["all_mesh_files"]

def render_mesh(test_num, mesh_dir, gen_idx, main_pose_sel_file_path, save_path, color_list, encode_mesh_path_list, mode='whole', sel_thresh=0.5, del_xml=True):
    # Note: 'cate' is inherently loaded from the global scope defined by argparse
    pose_sel_file_path = os.path.join(main_pose_sel_file_path, "{}_data_{}_{}.pt".format(cate, test_num, gen_idx))
    pose_sel = torch.load(pose_sel_file_path, map_location="cpu")

    if mode == 'whole':
        sel_vector_bool = pose_sel["sel_vector"] > sel_thresh
        sel_obj_list, _ = decode_mesh_code_list(encode_mesh_path_list[sel_vector_bool], mesh_dir)
        render_xml_path = "{}/render_{}_{}.xml".format(save_path, test_num, gen_idx)
        sel_color_list = [color_list[i] for i in range(len(color_list)) if sel_vector_bool[i]]
        save_render_xml(sel_obj_list, sel_color_list, render_xml_path)
        mitsuba_render(test_num, gen_idx, save_path, phi=45)
        mitsuba_render(test_num, gen_idx, save_path, phi=225)
        if del_xml:
            os.remove(render_xml_path)
    elif mode =='parts':
        sel_obj_list, _ = decode_mesh_code_list(encode_mesh_path_list, mesh_dir)
        for idx, obj_path in enumerate(sel_obj_list):
            render_xml_path = "{}/render_{}_{}.xml".format(save_path, test_num, idx)
            save_render_xml([obj_path], [color_list[idx]], render_xml_path)
            mitsuba_render(test_num, idx, save_path, phi=45)
            if del_xml:
                os.remove(render_xml_path)

def matching_find_all_idx_list(test_list, cate, main_pose_sel_file_path):
    all_idx_list = []
    for test_num in test_list:
        pattern = re.compile(r'{}_data_{}_([0-9]+).png'.format(cate, test_num))
        idx_list = []
        for file in os.listdir(main_pose_sel_file_path):
            match = pattern.match(file)
            if match:
                idx_list.append(int(match.group(1)))
        print("test_num: ", test_num, "all_idx_list: ", idx_list)
        all_idx_list.append(idx_list)
    return all_idx_list

# ==========================================
# Main Execution Block with Argparse
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Toolkit: Combine and render meshes using Mitsuba.")

    # List argument for multiple test indices
    parser.add_argument('--test_list', nargs='+', type=int, default=[0], 
                        help='List of test numbers (e.g., --test_list 0 22 115)')

    # Path arguments
    parser.add_argument('--mesh_dir', type=str, default="path/to/mesh_dir", 
                        help='Directory path for the dataset mesh')
    parser.add_argument('--data_dir', type=str, default="path/to/data_dir", 
                        help='Directory containing the .pt data files')
    parser.add_argument('--cate', type=str, default="category", 
                        help='The category of the mix set')
    parser.add_argument('--test_path', type=str, 
                        default="path/to/test_path", 
                        help='Path to the test output/checkpoints')
    parser.add_argument('--mesh_render_dir', type=str, default="path/to/mesh_render_dir", 
                        help='Output directory name for rendered meshes')

    # Boolean arguments with explicit toggles
    parser.add_argument('--need_gt_render', action='store_true', default=True, 
                        help='Enable rendering of GT shapes (default: True)')
    parser.add_argument('--no_gt_render', action='store_false', dest='need_gt_render', 
                        help='Disable rendering of GT shapes')

    parser.add_argument('--need_inputs', action='store_true', default=True, 
                        help='Enable rendering of input parts (default: True)')
    parser.add_argument('--no_inputs', action='store_false', dest='need_inputs', 
                        help='Disable rendering of input parts')

    args = parser.parse_args()

    # Map parsed arguments to global variables to retain original function logic without changes
    test_list = args.test_list
    mesh_dir = args.mesh_dir
    data_dir = args.data_dir
    cate = args.cate
    test_path = args.test_path
    need_gt_render = args.need_gt_render
    need_inputs = args.need_inputs
    mesh_render_dir = args.mesh_render_dir

    # ==========================================
    # Initialization and Processing
    # ==========================================
    dataset_path = "{}/{}/test".format(data_dir, cate)
    gen_main_pose_sel_file_path = "{}/gen_img".format(test_path)
    gt_main_pose_sel_file_path = "{}/ref_img".format(test_path)
    gen_save_path = "{}/{}/{}/gen".format(test_path, mesh_render_dir, cate)
    gt_save_path = "{}/{}/{}/ref".format(test_path, mesh_render_dir, cate)
    parts_save_path = "{}/{}/{}/parts".format(test_path, mesh_render_dir, cate)

    gen_all_idx_list = matching_find_all_idx_list(test_list, cate, gen_main_pose_sel_file_path)
    gt_all_idx_list = matching_find_all_idx_list(test_list, cate, gt_main_pose_sel_file_path)

    if not os.path.exists(gen_save_path):
        os.makedirs(gen_save_path)

    if not os.path.exists(gt_save_path):
        os.makedirs(gt_save_path)

    if not os.path.exists(parts_save_path):
        os.makedirs(parts_save_path)

    for global_idx in range(len(test_list)):
        color_list, encode_mesh_path_list = get_set_info(test_list[global_idx], dataset_path, cate)
        print("Your mesh_path_list: ", decode_mesh_code_list(encode_mesh_path_list)[1])
        # render gen
        for gen_idx in gen_all_idx_list[global_idx]:
            render_mesh(test_list[global_idx], mesh_dir, gen_idx, gen_main_pose_sel_file_path, gen_save_path, color_list, encode_mesh_path_list)
        
        if need_gt_render:
            # render gt
            for gt_idx in gt_all_idx_list[global_idx]:
                render_mesh(test_list[global_idx], mesh_dir, gt_idx, gt_main_pose_sel_file_path, gt_save_path, color_list, encode_mesh_path_list)
        
        if need_inputs:
            # render inputs
            render_mesh(test_list[global_idx], mesh_dir, 0, gen_main_pose_sel_file_path, parts_save_path, color_list, encode_mesh_path_list, mode='parts')
