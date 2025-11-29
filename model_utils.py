import onnx
import onnxruntime as ort
import numpy as np

def normalize_image_np(image, mean=0, std=1.0):
    max_pixel_value = image.max()
    return (image - mean * max_pixel_value) / (std * max_pixel_value)

def get_bounding_box(image, session):
    a = normalize_image_np(np.array(image))
    input = np.stack((a,a,a), axis=0)
    out = session.run(None, {'images': input[np.newaxis, :, :, :].astype(np.float32)})[0]

    idx = np.argmax(out[:, 4, :])

    bestBox = out[:, :, idx]

    bestBox = bestBox.flatten()

    x_center = bestBox[0]
    y_center = bestBox[1]
    diameter_x = bestBox[2]
    diameter_y = bestBox[3]

    return x_center, y_center, diameter_x, diameter_y


def load_onnx_model(path, device='cuda'):
    # Load the ONNX model
    model = onnx.load(path)
    
    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Create an ONNX Runtime session
    session = ort.InferenceSession(path, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])

    return session