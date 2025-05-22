# Summary

Contains all audio-encoder logic.

* **Train** – `align_audio_with_text_encoder.ipynb`
* **Infer** – `../tutorials/inference_examples.ipynb`

# Good to Know

* **Hardware used for training**
    * CPU    : Intel i9-14900KF @ 3.2 GHz
    * GPU    : RTX 4090, 24 GB VRAM
    * RAM    : 64 GB DDR5 (fully utilized)
    * Disk   : ≈ 1 TB cache (keeps I/O fast)
      Training ran for weeks and hadn’t fully converged. Pre-trained weights live in `../weights/pretrained.pth`.
* `align_audio_with_text_encoder` logs configs, metrics, debug images, and checkpoints to `training_logs/<run-id>/`.
* Tested with multiple CUDA 12.x releases; latest confirmed version is 12.8.
