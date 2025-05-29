# SNN-based-MNIST-classification

**INTRODUCTION**
Spike-driven neural networks, inspired by the brain’s event-driven processing, offer advantages like sparse data representation, efficient parallel processing, and low power consumption. These benefits make them ideal for MNIST classification, enabling faster and more accurate recognition of handwritten digits.
It enhances efficiency by providing non-volatile, high-density synaptic storage, making the system scalable for edge computing. The multi-head attention (MHA) mechanism, combined with a spiking-ANN hybrid model, allows the network to focus on different regions of digit images, capturing both global structures and fine details. This approach improves robustness against handwriting variations and distortions, ensuring efficient and adaptive digit classification.

**PROBLEM STATEMENT AND OBJECTIVE**
Traditional ANNs for digit recognition (e.g., MNIST) consume high power and perform dense computations.
SNNs offer energy-efficient processing but struggle with capturing spatial dependencies in images.
Multi-Head Attention (MHA) enhances feature extraction by focusing on different parts of an image simultaneously.
Combining MHA with SNNs improves accuracy, efficiency, and robustness against handwriting variations.
The goal is to achieve FASTER, LOW-POWER, AND ADAPTIVE DIGIT RECOGNITION compared to standard ANNs. 

**FLOWCHART**
[Pixel Intensities]
       ↓
  Rate Encoder
       ↓
  Spike Trains
       ↓
  Decay Filter (smoothing)
       ↓
  First Layer (8 → 2 neurons)
       ↓
  Membrane Potentials (pot0, pot1)
       ↓
Probabilistic Spike Generator
       ↓
  Output Spikes (classification)


  **LIST OF FILES**
  1) ALTERNATE PYTHOND CODE FOR VISUALIZING SPIKES AND MEMBRANE POTENTIAL
  2) PYTHON CODE FOR FILTERED SPIKE TRAIN
  3) PYTHON CODE FOR VISUALIZING PROBABILITY BASED OUTPUT SPIKING
  4) PYTHON CODE FOR VISUALIZING SPIKE TRAIN FOR ALL THE PIXELS
  5) PYTHON CODE FOR VISUALIZE RANDOM SPIKES
  6) PYTHON CODE FOR DESIGNING THE DECAY KERNEL
  7) PYHTON CODE FOR OUTPUT FILTERED SPIKE FEEDBACK
  8) MULTI-HEAD ATTENTION
  9) FULL CODE FOR CLASSIFICATION

**PACKAGES AND DEPENDENCIES**
| Import                   | Library            | Purpose                    |
| ------------------------ | ------------------ | -------------------------- |
| `torch`                  | PyTorch core       | Tensors, GPU computation   |
| `torch.nn`               | PyTorch NN module  | Layers, models             |
| `torch.optim`            | PyTorch Optimizers | SGD, Adam, etc.            |
| `torch.nn.functional`    | Functional ops     | Stateless functions        |
| `torchvision.datasets`   | Torchvision        | Datasets like MNIST        |
| `torchvision.transforms` | Torchvision        | Image preprocessing        |
| `torch.utils.data`       | PyTorch utils      | DataLoader, Subset         |
| `matplotlib.pyplot`      | Matplotlib         | Plotting and visualization |

