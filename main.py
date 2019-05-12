import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net


def run():
    use_cuda=True
    image_nc=1
    epochs = 60
    batch_size = 128
    BOX_MIN = 0
    BOX_MAX = 1

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model1 = "./MNIST_target_model_1.pth"
    targeted_model1 = MNIST_target_net().to(device)
    targeted_model1.load_state_dict(torch.load(pretrained_model1))
    targeted_model1.eval()

    pretrained_model2 = "./MNIST_target_model_2.pth"
    targeted_model2 = MNIST_target_net().to(device)
    targeted_model2.load_state_dict(torch.load(pretrained_model2))
    targeted_model2.eval()


    model_num_labels = 10

    # MNIST train dataset and dataloader declaration
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    advGAN = AdvGAN_Attack(device,
                            targeted_model1,
                            targeted_model2,
                            model_num_labels,
                            image_nc,
                            BOX_MIN,
                            BOX_MAX)

    advGAN.train(dataloader, epochs)



if __name__ == '__main__':
    run()