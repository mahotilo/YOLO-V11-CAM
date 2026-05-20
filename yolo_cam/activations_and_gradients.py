import torch

class ActivationsAndGradients:
  """ Class for extracting activations and
      registering gradients from targetted intermediate layers """

  def __init__(self, model, target_layers, reshape_transform):
    self.model = model
    self.gradients = []
    self.activations = []
    self.reshape_transform = reshape_transform
    self.handles = []
    for target_layer in target_layers:
      self.handles.append(
        target_layer.register_forward_hook(self.save_activation))
      # Because of https://github.com/pytorch/pytorch/issues/61519,
      # we don't use backward hook to record gradients.
      self.handles.append(
        target_layer.register_forward_hook(self.save_gradient))

  def save_activation(self, module, input, output):
    activation = output

    if self.reshape_transform is not None:
      activation = self.reshape_transform(activation)
    self.activations.append(activation.cpu().detach())

  def save_gradient(self, module, input, output):
    if not hasattr(output, "requires_grad") or not output.requires_grad:
      # You can only register hooks on tensor requires grad.
      return

    # Gradients are computed in reverse order
    def _store_grad(grad):
      if self.reshape_transform is not None:
        grad = self.reshape_transform(grad)
      self.gradients = [grad.cpu().detach()] + self.gradients

    output.register_hook(_store_grad)

  def __call__(self, x):
    self.gradients = []
    self.activations = []
    
    # Перевіряємо тип вхідних даних
    if isinstance(x, torch.Tensor):
      # Якщо це вже тензор, ми викликаємо внутрішню PyTorch-модель напрямую,
      # щоб градієнти EigenCAM знімалися безпосередньо з шарів компіляції.
      if hasattr(self.model, 'model') and hasattr(self.model.model, 'forward'):
        return self.model.model(x)
      return self.model(x)
    else:
      # Якщо це NumPy масив, викликаємо стандартний predict
      # але нам потрібно повернути саме СИРИЙ вихід моделі, а не об'єкт Results,
      # щоб логіка обчислення loss.backward() у base_cam.py була однаковою!
      outputs = self.model(x)
      
      # Якщо повернувся об'єкт Results від Ultralytics, витягуємо з нього сирі логіти
      if isinstance(outputs, list) and len(outputs) > 0 and hasattr(outputs[0], 'probs'):
        # Повертаємо тензор логітів, який очікує блок обчислення таргетів
        if outputs[0].probs is not None:
          return outputs[0].probs.data.unsqueeze(0)
      
      return outputs

  def release(self):
    for handle in self.handles:
      handle.remove()
