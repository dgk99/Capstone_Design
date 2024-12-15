import torch

def check_cuda_status():
    """
    Check if CUDA is available and display CUDA details.
    """
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(torch.cuda.current_device())}")
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    check_cuda_status()
