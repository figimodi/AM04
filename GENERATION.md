## Generation of color manipulated defects
#### python .\src\generate_color_transferred_images.py
1. Color manipulated defects are stored inside **/data/Defects**   
    **Example**:  
    Image0_CT_0_0.jpg  
    Image0_CT_0_1.jpg  
    Image0_CT_0_2.jpg  
    Image0_CT_0_3.jpg  
    ...  
    Image0_CT_0_7.jpg  
    ...
    Image3_CT_2_0.jpg  
    Image3_CT_2_1.jpg  
    Image3_CT_2_2.jpg  
    Image3_CT_2_3.jpg  
    ...  
    Image3_CT_2_7.jpg  
    **Meaning**:  
    DefectImageName_CT_ID_ProgressiveNumberColorTransferingTechnique.jpg  
    (CT: color transfering)  
    PK(DefectImageName, ID)  
2. Respective masks (combinations of original masks) are store inside **/data/DefectsMasks**
    **Example**:  
    Image0_PD_01_Horizontal.jpg  
    Image0_CB_3_Horizontal.jpg  
    **Meaning**:  
    DefectImageName_PDorCB_ID_DefectType.jpg (PD: part defect, CB: combination)  
    PK(DefectImageName, ID)  

## Generation of synthetic defects
#### pyton .\src\generate_synthetic_images.py 
1. Synthethic defects are stored inside **/data/SynthethicDefects**   
    **Example**:  
    Image1_Vertical_8.jpg  
    Image1_Vertical_9.jpg  
    Image3_Vertical_9.jpg  
    Image11_Spattering_4.jpg  
    Image10_Spattering_0.jpg  
    **Meaning**:  
    NoDefectImageBackground_DefectType_IDwithinDefectType.jpg  
    PK(DefectType, IDwithinDefectType)  
2. Respective masks are stored inside **/data/SynthethicDefectMasks**  
    **Example**:  
    Image1_Vertical_8_mask.jpg  
    Image1_Vertical_9_mask.jpg  
    Image2_Vertical_4_mask.jpg  
    Image7_Vertical_1_mask.jpg  
    Image8_Spattering_4_mask.jpg  
    Image9_Spattering_3_mask.jpg  
    **Meaning**:  
    NoDefectImageBackground_DefectType_IDwithinDefectType_mask.jpg  
    PK(DefectType, IDwithinDefectType)  