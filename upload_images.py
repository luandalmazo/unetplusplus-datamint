
from datamint import Api
from datamint.mlflow import set_project
from progress.bar import Bar
import os
from tqdm import tqdm

PROJECT_NAME = "TMJ Test"
DATA_DIR = "output_nifti"
SEG_DIR = "output_nifti_seg"
should_upload_images = False 
CLASS_NAMES =   {1:"Disc", 2:"Condyle", 3:"Eminence"}

def main():
    print("Setting up project...")
    
    api = Api()
    proj = api.projects.get_by_name(PROJECT_NAME)
    
    print(f"Using existing project '{PROJECT_NAME}'")

    set_project(PROJECT_NAME)

    ''' Get the lenght of files in DATA_DIR '''
    subdirs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith("_0000.nii.gz")]
    bar = Bar("Processing cases", max=len(subdirs))

    image_paths = []
    mask_paths = []

    for case_dir in subdirs:
        case_id = os.path.basename(case_dir)
        img_path = os.path.join(DATA_DIR, f"{case_id}")
        
        seg_case_id = case_id.replace("_0000", "")
        mask_path = os.path.join(SEG_DIR, f"{seg_case_id}")
        
        image_paths.append(img_path)
        mask_paths.append(mask_path)

        bar.next()

    bar.finish()
    
    print("Found", len(image_paths))
    
    # -------------------------------
    # Upload images as resources
    # -------------------------------  
    
    print(f"Should upload images: {should_upload_images}")
    
    if should_upload_images:
        print("Uploading image volumes to Datamint...")  
        uploaded_resources = api.resources.upload_resources(
        [str(p) for p in image_paths],
            tags=['mri', 'tmj', '2022'],
            publish_to=proj,
            progress_bar=True
        )
        print(f"Uploaded {len(uploaded_resources)} images to Datamint")
 
    # -------------------------------
    # Upload masks as segmentations
    # -------------------------------
    print("Uploading segmentation masks...")
    all_resources = list(api.resources.get_list(project_name=PROJECT_NAME))
    filename_to_resource = {r.filename: r for r in all_resources}
    
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths)):
        resource = filename_to_resource[os.path.basename(img_path)]
        
        api.annotations.upload_segmentations(
            resource=resource,
            file_path=mask_path,
            imported_from="LabelSys",
            name=CLASS_NAMES
        )

    print("Segmentation masks uploaded successfully!")
    
    print("All done.")

if __name__ == "__main__":
    main()