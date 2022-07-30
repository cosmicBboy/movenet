from torchtyping import TensorType


AudioTensor = TensorType["batch", "channels", "frames"]
VideoTensor = TensorType["batch", "frames", "height", "width", "channels"]
