from torch.utils.cpp_extension import load
chamfer_3D = load(name="chamfer_3D",
sources=[
"/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer_cuda.cpp"]),
"/".join(os.path.abspath(__file__).split('/')[:-1] + ["chamfer3D.cu"]),
])