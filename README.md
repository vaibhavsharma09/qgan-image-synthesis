# Quantum Generative Models for Image Synthesis

A Quantum Generative Adversarial Network (QGAN) built for the Quantum Machine Learning course in my M.Sc. in Artificial Intelligence at Universität zu Lübeck.

A 4-qubit parameterized quantum circuit (PQC) is trained as the generator to learn the distribution of handwritten digit 5 at two resolutions, 8x8 and 16x16, while keeping the adversarial training stable. A classical GAN is included as a baseline for comparison.

## What is in this repo

| File | Contents |
|---|---|
| `qgan_and_classical_gan_baseline.ipynb` | Full notebook: classical GAN baseline, QGAN experiments, evaluation |
| `colab_notebook_export.pdf` | PDF export of the Colab notebook with all outputs |
| `qgan_project_report.doc` | Project report |
| `project_submission_slides.ppt` / `.pdf` | Presentation slides |
| `demo_video_1.mp4`, `demo_video_2.mp4` | Recorded walkthroughs of the training runs |

The original notebook is also on [Google Colab](https://colab.research.google.com/drive/1qGn56cnEjr8Gp7ODe4VWAZLUrlsSuHr5?usp=sharing).

## Approach

- **Mean-pool amplitude mapping.** The 256-pixel (16x16) and 64-pixel (8x8) images are reduced so they fit a 4-qubit circuit.
- **Latent to circuit.** A latent vector is mapped to rotation angles with `tanh(z · W) · π` and bound to the PQC.
- **Stability.** Label smoothing and Gaussian latent sampling are used to prevent mode collapse.
- **Data pipeline.** A custom loader handles the string headers in the 8x8 CSV dataset.

## Experiments

1. **16x16:** higher-resolution scaling using mean-pool mapping.
2. **8x8:** faster convergence with a cleaner feature set.

The evaluation cells produce a state fidelity analysis and image grids. The target state fidelity is above 0.95.

## Running it

```bash
pip install qiskit qiskit-machine-learning torch numpy pandas matplotlib gdown
```

Open the notebook in Colab or Jupyter and run the cells in order. The random seed is fixed at 42. The datasets are downloaded automatically with `gdown`.

## Author

Vaibhav Sharma, M.Sc. Artificial Intelligence, Universität zu Lübeck
