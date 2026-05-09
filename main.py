import torch
def main():
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.backends.cuda.is_built())
    print("Hello from til-26-overflow!")


if __name__ == "__main__":
    main()
