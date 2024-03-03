import torch
import torchvision
import torchvision.transforms as transforms
import os

# Assume Net class is defined here or imported

def evaluate_models(model_dir='./model'):
    best_model = None
    best_accuracy = 0

    # Load and preprocess CIFAR-10 test data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    for model_name in os.listdir(model_dir):
        if model_name == "best_model.pt":
            continue
        model_path = model_dir + '/' + model_name
        net = torch.jit.load(model_path)
        net.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Model: {model_name}, Accuracy: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name

    # Save best model info
    torch.save(best_model, './model/best_model.pt')

    return best_model, best_accuracy

if __name__ == "__main__":
    best_model, best_accuracy = evaluate_models()
    print(f'Best Model: {best_model} with accuracy {best_accuracy}')
