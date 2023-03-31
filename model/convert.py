import onnx
import torch
from onnx2pytorch import ConvertModel
import tf2onnx
# not included in requirements.txt because there is a version conflict with tensorflow

def convert_onnx_to_pytorch(onnx_model_path, pytorch_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Convert the ONNX model to PyTorch
    pytorch_model = ConvertModel(onnx_model)

    # Save the PyTorch model
    torch.save(pytorch_model.state_dict(), pytorch_model_path)
    

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print(f"Usage: {sys.argv[0]} <tensorflow_model_path> <pytorch_model_path>")
    #     sys.exit(1)
    
    if len(sys.argv) == 3
        onnx_model_path = sys.argv[1]
        pytorch_model_path = sys.argv[2]
    else:
        onnx_model_path = 'model.onnx'
        pytorch_model_path = 'model.pt'
    
    convert_onnx_to_pytorch(onnx_model_path, pytorch_model_path)