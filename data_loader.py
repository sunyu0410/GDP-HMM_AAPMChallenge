

'''
This script is adapted from the data loader of below CVPR paper. 
If you find the functions in this script are helpful to you (for the challenge and beyond), please kindly cite the original paper: 

Riqiang Gao, Bin Lou, Zhoubing Xu, Dorin Comaniciu, and Ali Kamen. 
"Flexible-cm gan: Towards precise 3d dose prediction in radiotherapy." 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
'''

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
import pdb

from scipy import ndimage 
from toolkit import *


HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",  "GreatVessels", "Trachea", "Body_Ring0-3"]

Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}



class MyDataset(Dataset):
    
    def __init__(self, cfig, phase):
        '''
        phase: train, validation, or testing 
        
        cfig: the configuration dictionary
        
            train_bs: training batch size
            val_bs: validation batch size
            num_workers: the number of workers when call the DataLoader of PyTorch
            
            csv_root: the meta data file, include patient id, plan id, the .npz data path and some conditions of the plan. 
            scale_dose_dict: path of a dictionary. The dictionary includes the prescribed doses of the PTVs. 
            pat_obj_dict: path of a dictionary. The dictionary includes the ROIs (PTVs and OARs) names used in optimization. 
            
            down_HU: bottom clip of the CT HU value. 
            up_HU: upper clip of the CT HU value. 
            denom_norm_HU: the denominator when normalizing the CT. 
            
            in_size & out_size: the size parameters used in data transformation. 

            norm_oar: True or False. Normalize the OAR channel or not. 
            CatStructures: True or False. Concat the PTVs and OARs in multiple channels, or merge them in one channel, respectively. 

            dose_div_factor: the value used to normalize dose. 
            
        '''
        
        self.cfig = cfig
        
        df = pd.read_csv(cfig['csv_root'])
        
        # df = df.loc[df['phase'] == phase]
        df = df.loc[df['dev_split'] == phase]

        # for debug 
        # df = df.loc[df['site'] == 1]
        # df = df.loc[df['isVMAT'] == True]

        self.phase = phase
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        ID = self.data_list[index].split('/')[-1].replace('.npz', '')
        PatientID = ID.split('+')[0]

        if len(str(PatientID)) < 3:
            PatientID = f"{PatientID:0>3}"
        
        data_npz = np.load(data_path, allow_pickle=True)

        In_dict = dict(data_npz)['arr_0'].item()

        In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] 

        ori_img_size = In_dict['img'].shape
        
        if 'dose' in In_dict.keys():
            ptv_highdose =  self.scale_dose_Dict[PatientID]['PTV_High']['PDose']
            In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale'] 
            PTVHighOPT = self.scale_dose_Dict[PatientID]['PTV_High']['OPTName']
            norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict[PTVHighOPT].astype('bool')], 3) + 1e-5) # D97
            In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor']
            In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2)


        isocenter = In_dict['isocenter']

        angle_plate_3D = np.zeros(In_dict['img'].shape)

        z_begin = int(isocenter[0]) - 5
        z_end = int(isocenter[0]) + 5
        z_begin = max(0, z_begin)
        z_end = min(angle_plate_3D.shape[0], z_end)
        
        if 'angle_plate' not in In_dict.keys():
            print (' **************** angle_plate error in ', data_path)
            In_dict['angle_plate'] = np.ones(In_dict['img'][0].shape)

        D3_plate = np.repeat(In_dict['angle_plate'][np.newaxis, :, :], z_end - z_begin, axis = 0)
        
        if D3_plate.shape[1] != angle_plate_3D.shape[1] or D3_plate.shape[2] != angle_plate_3D.shape[2]:
            D3_plate = ndimage.zoom(D3_plate, (1, angle_plate_3D.shape[1] / D3_plate.shape[1], angle_plate_3D.shape[2] / D3_plate.shape[2]), order = 0)
        angle_plate_3D[z_begin: z_end] = D3_plate

        In_dict['angle_plate'] = angle_plate_3D
        
        KEYS = list(In_dict.keys())
        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
            else:
                KEYS.remove(key)
        
        if self.phase == 'train':
            if 'with_aug' in self.cfig.keys() and not self.cfig['with_aug']:
                self.aug = tt_augmentation(KEYS, self.cfig['in_size'],  self.cfig['out_size'], isocenter)
            else:
                self.aug = tr_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)
            
        if self.phase in ['val', 'test', 'valid', 'external_test']:
            self.aug = tt_augmentation(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)


        In_dict = self.aug(In_dict)

        if self.site_list[index] > 1.5:
            OAR_LIST = HaN_OAR_LIST
            OAR_DICT = HaN_OAR_DICT
        else:
            OAR_LIST = Lung_OAR_LIST
            OAR_DICT = Lung_OAR_DICT

        data_dict = dict()  

        try:
            need_list = self.pat_obj_dict[ID.split('+')[0]] # list(a.values())[0]
        except:
            need_list = OAR_LIST
            print (ID.split('+')[0],  '-------------not in the pat_obj_dict')
        comb_oar, cat_oar  = combine_oar(In_dict, need_list, self.cfig['norm_oar'], OAR_DICT)

        opt_dose_dict = {}
        dose_dict = {}
        for key in self.scale_dose_Dict[PatientID].keys():
            if key in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
                opt_dose_dict[self.scale_dose_Dict[PatientID][key]['OPTName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']
                if key != 'PTV_High':
                    dose_dict[self.scale_dose_Dict[PatientID][key]['StructName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']
                else:
                    dose_dict[self.scale_dose_Dict[PatientID][key]['OPTName']] = self.scale_dose_Dict[PatientID][key]['PDose'] / self.cfig['dose_div_factor']
                
            
        comb_optptv, prs_opt, cat_optptv = combine_ptv(In_dict, opt_dose_dict)
        comb_ptv, _, cat_ptv  = combine_ptv(In_dict, dose_dict)

        

        data_dict['angle_plate'] = In_dict['angle_plate']
        data_dict['beam_plate'] = In_dict['beam_plate']

        if 'dose' in In_dict.keys():
            data_dict['label'] = In_dict['dose'] * In_dict['Body'] 

        data_dict['prompt'] = torch.tensor([In_dict['isVMAT'], len(prs_opt), self.site_list[index], self.cohort_list[index]]).float()
        
        prompt_extend = data_dict['prompt'][None, :, None, None].repeat(1, self.cfig['out_size'][0] // 4, self.cfig['out_size'][1], self.cfig['out_size'][2])
        
        if self.cfig['CatStructures']:
            data_dict['data'] = torch.cat((cat_optptv, cat_ptv, cat_oar, In_dict['Body'], In_dict['img'], data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0)
        else:
            data_dict['data'] = torch.cat((comb_optptv, comb_ptv, comb_oar, In_dict['Body'], In_dict['img'], data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0) # , In_dict['obj_2DGy'], In_dict['obj_2DWei']
        

        data_dict['oar'] = cat_oar
        data_dict['ptv'] = cat_ptv
        data_dict['optptv'] = cat_optptv

        prescribed_dose = [self.scale_dose_Dict[PatientID]['PTV_High']['PDose']]
        if 'PTV_Mid' in self.scale_dose_Dict[PatientID].keys():
            prescribed_dose.append(self.scale_dose_Dict[PatientID]['PTV_Mid']['PDose'])
        else:
            prescribed_dose.append(0)
        if 'PTV_Low' in self.scale_dose_Dict[PatientID].keys():
            prescribed_dose.append(self.scale_dose_Dict[PatientID]['PTV_Low']['PDose'])
        else:
            prescribed_dose.append(0)

        data_dict['prescrbed_dose'] = torch.tensor(prescribed_dose).float()

        del In_dict

        data_dict['id'] = ID
        data_dict['ori_img_size'] = torch.tensor(ori_img_size)
        data_dict['ori_isocenter'] = torch.tensor(isocenter)

        return data_dict
    
class GetLoader(object):
    def __init__(self, cfig):
        super().__init__()
        self.cfig = cfig
        
    def train_dataloader(self):
        dataset = MyDataset(self.cfig,  phase='train') 
        return DataLoader(dataset, batch_size=self.cfig['train_bs'],  shuffle=True, num_workers=self.cfig['num_workers'])
    
    def val_dataloader(self):
        dataset = MyDataset(self.cfig, phase='valid') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])
    
    def test_dataloader(self):
        dataset = MyDataset(self.cfig, phase='test') 
        return DataLoader(dataset, batch_size=self.cfig['val_bs'], shuffle=False, num_workers=self.cfig['num_workers'])


class InputDataset(Dataset):
    
    def __init__(self, cfig, phase):
        '''
        phase: train, validation, or testing 
        
        cfig: the configuration dictionary
        
            train_bs: training batch size
            val_bs: validation batch size
            num_workers: the number of workers when call the DataLoader of PyTorch
            
            csv_root: the meta data file, include patient id, plan id, the .npz data path and some conditions of the plan. 
            scale_dose_dict: path of a dictionary. The dictionary includes the prescribed doses of the PTVs. 
            pat_obj_dict: path of a dictionary. The dictionary includes the ROIs (PTVs and OARs) names used in optimization. 
            
            down_HU: bottom clip of the CT HU value. 
            up_HU: upper clip of the CT HU value. 
            denom_norm_HU: the denominator when normalizing the CT. 
            
            in_size & out_size: the size parameters used in data transformation. 

            norm_oar: True or False. Normalize the OAR channel or not. 
            CatStructures: True or False. Concat the PTVs and OARs in multiple channels, or merge them in one channel, respectively. 

            dose_div_factor: the value used to normalize dose. 
            
        '''
        
        self.cfig = cfig
        
        df = pd.read_csv(cfig['csv_root'])
        
        df = df.loc[df['dev_split'] == phase]

        self.phase = phase
        self.data_list = df['npz_path'].tolist()
        self.site_list = df['site'].tolist()
        self.cohort_list = df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))

        self.dose_div_factor = self.cfig['dose_div_factor']

    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def make_3d_angle_plate(plate_2d, shape, isocenter):
        plate = np.zeros(shape)

        z_begin = int(isocenter[0]) - 5
        z_end = int(isocenter[0]) + 5
        z_begin = max(0, z_begin)
        z_end = min(plate.shape[0], z_end)
        
        slices = np.repeat(plate_2d[np.newaxis, :, :], z_end - z_begin, axis = 0)
        
        if slices.shape[1:] != plate.shape[1:]:
            slices = ndimage.zoom(
                slices, 
                (1, plate.shape[1] / slices.shape[1], plate.shape[2] / slices.shape[2]), 
                order = 0
            )

        plate[z_begin: z_end] = slices
        return plate

    def __getitem__(self, idx):
        # Patient ID
        data_path = self.data_list[idx]
        _id = self.data_list[idx].split('/')[-1].replace('.npz', '') # 0617-259694+imrt+MOS_33896
        pat_id = _id.split('+')[0] # 0617-259694
        if len(str(pat_id)) < 3: pat_id = f"{pat_id:0>3}"

        # Read data
        data_npz = np.load(data_path, allow_pickle=True)
        npy_data = dict(data_npz)['arr_0'].item()

        # CT
        ct = np.clip(npy_data['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] 
        ori_img_size = ct.shape

        # Dose 
        ptv_info = self.scale_dose_Dict[pat_id]
        if 'dose' in npy_data.keys():
            ptv_highdose =  ptv_info['PTV_High']['PDose']
            ptv_high_name = ptv_info['PTV_High']['OPTName']

            dose = npy_data['dose'] * npy_data['dose_scale'] 

            norm_scale = ptv_highdose / (np.percentile(dose[npy_data[ptv_high_name].astype('bool')], 3) + 1e-5) # D97
            dose = dose * norm_scale / self.dose_div_factor
            dose = np.clip(dose, 0, ptv_highdose * 1.2)
            npy_data['dose'] = dose

        # Isocenter
        isocenter = npy_data['isocenter']

        # Angle plate
        plate_3d = self.make_3d_angle_plate(npy_data['angle_plate'], ct.shape, isocenter)

        # Reshape (and augment) the data
        keys = list(npy_data.keys())
        for key, value in npy_data.items(): 
            if isinstance(value, np.ndarray) and len(value.shape) == 3:
                npy_data[key] = torch.from_numpy(value.astype('float'))[None]
            else:
                keys.remove(key)

        assert self.phase in ('train', 'val', 'test', 'valid', 'external_test')

        if self.phase=='train' and self.cfig.get('with_aug'):
            aug_func = tr_augmentation
        else:
            aug_func = tt_augmentation
        self.aug = aug_func(keys, self.cfig['in_size'], self.cfig['out_size'], isocenter)
        npy_data = self.aug(npy_data)

        # OAR 
        OAR_LIST, OAR_DICT = (HaN_OAR_LIST, HaN_OAR_DICT) if self.site_list[idx] == 2 else \
                                (Lung_OAR_LIST, Lung_OAR_DICT)

        # Prepare output
        data_dict = dict()  

        assert pat_id in self.pat_obj_dict
        need_list = self.pat_obj_dict[pat_id]

        comb_oar, cat_oar  = combine_oar(npy_data, need_list, self.cfig['norm_oar'], OAR_DICT)

        opt_dose_dict = {}
        dose_dict = {}
        

        for key in ptv_info:
            if key not in ['PTV_High', 'PTV_Mid', 'PTV_Low']: continue
            
            
            opt_name = ptv_info[key]['OPTName']
            opt_dose_dict[opt_name] = ptv_info[key]['PDose'] / self.dose_div_factor

            if key != 'PTV_High':
                struct_name = ptv_info[key]['StructName']
                dose_dict[struct_name] = ptv_info[key]['PDose'] / self.dose_div_factor
            else:
                dose_dict[opt_name] = ptv_info[key]['PDose'] / self.dose_div_factor
                
        
        # opt: the labels for optimisation
        comb_optptv, prs_opt, cat_optptv = combine_ptv(npy_data, opt_dose_dict)
        comb_ptv, _, cat_ptv  = combine_ptv(npy_data, dose_dict)

        data_dict['angle_plate'] = plate_3d
        data_dict['beam_plate'] = npy_data['beam_plate']

        data_dict['label'] = npy_data['dose'] * npy_data['Body'] 

        data_dict['prompt'] = torch.tensor([npy_data['isVMAT'], len(prs_opt), self.site_list[idx], self.cohort_list[idx]]).float() # [False, 3, 2, 1]
        
        prompt_extend = data_dict['prompt'][None, :, None, None].repeat(1, self.cfig['out_size'][0] // 4, self.cfig['out_size'][1], self.cfig['out_size'][2])
        
        if self.cfig['CatStructures']:
            data_dict['data'] = torch.cat((cat_optptv, cat_ptv, cat_oar, npy_data['Body'], ct, data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0)
        else:
            data_dict['data'] = torch.cat((comb_optptv, comb_ptv, comb_oar, npy_data['Body'], ct, data_dict['beam_plate'], data_dict['angle_plate'], prompt_extend), axis=0) 

        data_dict['oar'] = cat_oar
        data_dict['ptv'] = cat_ptv
        data_dict['optptv'] = cat_optptv

        prescribed_dose = [
            ptv_info['PTV_High']['PDose'],
            ptv_info['PTV_Mid']['PDose'] if 'PTV_Mid' in ptv_info else 0,
            ptv_info['PTV_Mid']['PDose'] if 'PTV_Low' in ptv_info else 0
        ]

        data_dict['prescrbed_dose'] = torch.tensor(prescribed_dose).float() # tensor([60.,  0.,  0.])

        data_dict['id'] = _id
        data_dict['ori_img_size'] = torch.tensor(ori_img_size)
        data_dict['ori_isocenter'] = torch.tensor(isocenter)

        return data_dict
    
if __name__ == '__main__':

    cfig = {
            'train_bs': 2,
             'val_bs': 2, 
             'num_workers': 2, 
             'csv_root': 'meta_files/meta_data.csv',
             'scale_dose_dict': 'meta_files/PTV_DICT.json',
             'pat_obj_dict': 'meta_files/Pat_Obj_DICT.json',
             'down_HU': -1000,
             'up_HU': 1000,
             'denom_norm_HU': 500,
             'in_size': (96, 128, 144), 
             'out_size': (96, 128, 144), 
             'norm_oar': True,
             'CatStructures': False,
             'dose_div_factor': 10 
             }
    
    loaders = GetLoader(cfig)
    train_loader = loaders.train_dataloader()

    for i, data in enumerate(train_loader):

        pdb.set_trace()
        print (data['data'].shape, data['label'].shape, data['beam_plate'].shape, data['prompt'].shape, data['ori_img_size'])

        
    
