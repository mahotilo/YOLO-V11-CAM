import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from yolo_cam.activations_and_gradients import ActivationsAndGradients
from yolo_cam.utils.svd_on_activations import get_2d_projection
from yolo_cam.utils.image import scale_cam_image
from yolo_cam.utils.model_targets import ClassifierOutputTarget

class BaseCAM:
  def __init__(self,
               model: torch.nn.Module,
               target_layers: List[torch.nn.Module],
               task: str = 'od',
               reshape_transform: Callable = None,
               compute_input_gradient: bool = False,
               uses_gradients: bool = True) -> None:
    self.model = model
    self.target_layers = target_layers
    self.task = task
    self.reshape_transform = reshape_transform
    self.compute_input_gradient = compute_input_gradient
    self.uses_gradients = uses_gradients
    self.activations_and_grads = ActivationsAndGradients(
      self.model, target_layers, reshape_transform)
    self.outputs = []
        
  """ Get a vector of weights for every channel in the target layer.
      Methods that return weights channels,
      will typically need to only implement this function. """

  def get_cam_weights(self,
                      input_tensor: np.array,
                      target_layers: List[torch.nn.Module],
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor) -> np.ndarray:
    raise Exception("Not Implemented")

  def get_cam_image(self,
                    input_tensor: np.array,
                    target_layer: torch.nn.Module,
                    targets: List[torch.nn.Module],
                    activations: torch.Tensor,
                    grads: torch.Tensor,
                    eigen_smooth: bool = False) -> np.ndarray:

    weights = self.get_cam_weights(input_tensor,
                                   target_layer,
                                   targets,
                                   activations,
                                   grads)
    weighted_activations = weights[:, :, None, None] * activations
    if eigen_smooth:
      cam = get_2d_projection(weighted_activations)
    else:
      cam = weighted_activations.sum(axis=1)
    return cam

  def forward(self,
              input_tensor: torch.Tensor,
              targets: List[torch.nn.Module],
              eigen_smooth: bool = False) -> np.ndarray:

    # Визначаємо, як виконувати інференс моделі
    if isinstance(input_tensor, torch.Tensor):
      # Передаємо тензор у predict, щоб Ultralytics зібрав повноцінний Results
      # verbose=False вимикає зайвий спам у консолі
      results = self.model.predict(input_tensor, verbose=False)
      # Зберігаємо об'єкт Results прямо в екземпляр класу CAM
      self.inference_result = results[0] if isinstance(results, list) else results
            
      # Оскільки activations_and_grads знімає хуки при прямому прогоні, 
      # викликаємо його, але вихід ігноруємо, бо результати вже маємо
      _ = self.activations_and_grads(input_tensor)
    else:
      # Якщо прийшов NumPy (старий шлях)
      results = self.activations_and_grads(input_tensor)
      self.inference_result = results[0] if isinstance(results, list) else results

    # Записуємо у стандартний список для сумісності з вашими старими викликами
    self.outputs = [self.inference_result]

    if targets is None:
      if self.task == 'cls':
        # Тепер об'єкт Results гарантовано тут є, і .probs.top5 працює завжди!
        target_categories = self.inference_result.probs.top5
      elif self.task == 'od':
        target_categories = self.inference_result.boxes.cls
      elif self.task == 'seg':
        target_categories = [category['name'] for category in self.inference_result.summary()]
      else:
        print('Invalid Task Entered')
                
      targets = [ClassifierOutputTarget(category) for category in target_categories]

    cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
    return self.aggregate_multi_layers(cam_per_layer)

  def get_target_width_height(self,
                              input_tensor: np.array) -> Tuple[int, int]:
    if isinstance(input_tensor, torch.Tensor):
      height, width = input_tensor.shape[-2], input_tensor.shape[-1]
    else:
      height, width = np.shape(input_tensor)[0], np.shape(input_tensor)[1]
    return width, height

  def compute_cam_per_layer(
          self,
          input_tensor: np.array,
          targets: List[torch.nn.Module],
          eigen_smooth: bool) -> np.ndarray:
    activations_list = [a.cpu().data.numpy()
                        for a in self.activations_and_grads.activations]
    grads_list = [g.cpu().data.numpy()
                  for g in self.activations_and_grads.gradients]
    target_size = self.get_target_width_height(input_tensor)

    cam_per_target_layer = []
    for i in range(len(self.target_layers)):
      target_layer = self.target_layers[i]
      layer_activations = None
      layer_grads = None
      if i < len(activations_list):
        layer_activations = activations_list[i]
      if i < len(grads_list):
        layer_grads = grads_list[i]

      cam = self.get_cam_image(input_tensor,
                               target_layer,
                               targets,
                               layer_activations,
                               layer_grads,
                               eigen_smooth)
      cam = np.maximum(cam, 0)
      scaled = scale_cam_image(cam, target_size)
      cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer

  def aggregate_multi_layers(
          self,
          cam_per_target_layer: np.ndarray) -> np.ndarray:
    cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
    cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
    result = np.mean(cam_per_target_layer, axis=1)
    W, H = self.get_target_width_height(self._input_tensor)
    return scale_cam_image(result, target_size=(W, H))

  def forward_augmentation_smoothing(self,
                                     input_tensor: np.array,
                                     targets: List[torch.nn.Module],
                                     eigen_smooth: bool = False) -> np.ndarray:
    transforms = tta.Compose(
      [
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1, 1.1]),
      ]
    )
    cams = []
    for transform in transforms:
      augmented_tensor = transform.augment_image(input_tensor)
      cam = self.forward(augmented_tensor,
                         targets,
                         eigen_smooth)

      cam = cam[:, None, :, :]
      cam = torch.from_numpy(cam)
      cam = transform.deaugment_mask(cam)

      cam = cam.numpy()
      cam = cam[:, 0, :, :]
      cams.append(cam)

    cam = np.mean(np.float32(cams), axis=0)
    return cam

  def __call__(self,
               input_tensor: np.array,
               targets: List[torch.nn.Module] = None,
               aug_smooth: bool = False,
               eigen_smooth: bool = False) -> np.ndarray:
    self._input_tensor = input_tensor
    if aug_smooth is True:
      return self.forward_augmentation_smoothing(
        input_tensor, targets, eigen_smooth)

    return self.forward(input_tensor,
                        targets, eigen_smooth)

  def __del__(self):
    self.activations_and_grads.release()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    self.activations_and_grads.release()
    if isinstance(exc_value, IndexError):
      print(
        f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
      return True
