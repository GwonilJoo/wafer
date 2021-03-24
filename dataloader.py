from utils import *


data_transforms = {
    'train': transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.RandomRotation(np.random.randint(0,360), expand = True),
        transforms.Resize([224, 224]),
        
        
        
        #transforms.RandomHorizontalFlip(),
        #transforms.Resize([26,26]),
        #transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

def get_all_data(data_dir, mode):
    all_data = sorted(glob.glob(os.path.join(data_dir, mode, '*', '*')))

    #all_label = [names.index(data.split('\\')[-2]) for data in all_data]
    all_label = [names.index(data.split('/')[-2]) for data in all_data]
    return all_data, all_label
