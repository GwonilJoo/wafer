from dataloader import *
from model import *

def train_model(model, train_loader, device, args):
    print("Learning...\n")

    model = model.to(device)
    #model.apply(init_weights)
    model.load_state_dict(torch.load('./saved/resnet-pretrained_1e-2/9_model.pt'))
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #best_acc = 0.0
    
    for epoch in range(args.num_epochs):
        avg_loss = 0
        total = 0
        correct = 0
        
        model.train()
        for x,y in tqdm(train_loader, desc="[Epoch:{}/{}]".format(epoch+1, args.num_epochs)):
            x = x.to(device)
            y = y.to(device)
        
            pred = model(x)
            loss = criterion(pred, y)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            avg_loss += loss / len(train_loader)
            correct += (y == torch.argmax(pred, 1)).float().sum().item()
            total += len(y)
    
        print("Avg loss = {}".format(avg_loss))
        print("Accuracy = {}".format(correct/total))
        
        output_path = args.save_dir + str(epoch) + "_model.pt"
        torch.save(model.state_dict(), output_path)

    
    print("Learning Finished!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='cls',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./dataset/train', type=str)
    parser.add_argument('--save_dir', type=str, default='./saved/resnet-pretrained_1e-3/')
    parser.add_argument('--random_seed', type=int, default='222')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)


    set_seed(device, args.random_seed)

    train_dataset = torchvision.datasets.ImageFolder(root = args.data_dir, transform = data_transforms['train'])
  
    train_loader = DataLoader(train_dataset, 
    			batch_size=args.batch_size, 
                      	shuffle=False, 
                      	drop_last=False, 
                      	sampler=weight_sampler(train_dataset)
                      	)
    
#        valid_loader = DataLoader(valid_dataset, 
#                          batch_size=args.batch_size, 
#                          shuffle=False, 
#                          drop_last=False, 
#                          sampler=weight_sampler(valid_dataset))

    train_sample = [0]*9
    
    for data,label in tqdm(train_loader, desc="check dataset uniform"):
        for i in range(9):
            train_sample[i] += torch.sum(label==i)
            
#        for data,label in valid_loader:
#            for i in range(9):
#                valid_sample[i] += torch.sum(label==i)
            
    print("train_sample: ", train_sample)
#        print("valid_sample: ", valid_sample)

    # model
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    train_model(model, train_loader, device, args)
