from process_heatmap import generate_combined_click_heatmap, save_combined_click_heatmap, parse_click_json
import os
import json

if __name__=="__main__":
    input_path = '/blue/kgong/yxing2/AutoPET/FDG_PSMA_PETCT_pre-simulated_clicks'
    output_path = '/blue/kgong/yxing2/AutoPET2025/pet_ct/pet_ct/nnUNet_raw/Dataset140_Autopet4/imagesTr'
    pet_path = '/blue/kgong/yxing2/AutoPET2025/pet_ct/pet_ct/nnUNet_raw/Dataset130_Autopet3/imagesTr'
    click_volumes = [v for v in os.listdir(input_path) if v.split('.')[-1] =='json']
    print("click_volumes",click_volumes[0])
    for i in click_volumes:
        volume_path = os.path.join(input_path, i)
        input_pet_name = f'{i.split("_clicks.json")[0]}_0000.nii.gz'
        input_pet_path = os.path.join(pet_path, input_pet_name)
        with open(volume_path, 'r') as file:
            json_data = json.load(file)
        clicks = parse_click_json(json_data)
        save_combined_click_heatmap(clicks,output_path, input_pet_path)
    print("process complete")
