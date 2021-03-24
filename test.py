from dataloader import *
from model import *

def test_model(model, test_loader, device, weight_path):
    print("Testing...\n")

    model = model.to(device)
    model.load_state_dict(torch.load(weight_path))

    model.eval()
    with torch.no_grad():
        total = [0]*9
        correct = [0]*9
        acc = [0]*9
        
        all_total, all_correct = 0, 0
        for x,y in tqdm(test_loader, desc="Test"):
            x = x.to(device)
            y = y.to(device)
        
            pred = model(x)
            
            all_total += len(y)
            all_correct += (y==torch.argmax(pred,1)).float().sum().item()
            
            for i in range(len(y)):
            	total[int(y[i])] += 1
            	correct[int(y[i])] += (y[i] == torch.argmax(pred, 1)[i]).float().item()
            	
        acc = [correct[i]/total[i]*100 for i in range(9)]
        
        for i in range(9):
        	print("{}:{}/{}, {}%".format(names[i], correct[i], total[i], acc[i]))	
        #print("Class Accuracy: ", acc)
        print("Total Accuracy: ", all_correct/all_total*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='cls',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./dataset/test', type=str)
    parser.add_argument('--weight_path', default='./saved/resnet-pretrained_1e-3/9_model.pt', type=str)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # runner_name = os.path.basename(__file__).split(".")[0] 
    # model_dir= args.exp_root + '{}'.format(runner_name)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # args.model_dir = model_dir+'/'+args.model_name+'.pth'


    test_dataset = torchvision.datasets.ImageFolder(root = args.data_dir, transform = data_transforms['test'])
    test_loader = DataLoader(test_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        drop_last=False, 
                    )
    
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    test_model(model, test_loader, device, args.weight_path)
