import torch
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from stable_diffusion_tensorrt_txt2img import  TensorRTStableDiffusionPipeline

# Use the DDIMScheduler scheduler here instead
scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler")

pipe = TensorRTStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                               revision='fp16', torch_dtype=torch.float16, scheduler=scheduler,)

# re-use cached folder to save ONNX models and TensorRT Engines
pipe.set_cached_folder("stabilityai/stable-diffusion-2-1", revision='fp16',)
pipe = pipe.to("cuda")

def run(pipe, batch):
  prompts = ["a beautiful photograph of Mt. Fuji during cherry blossom", "flower", "tree", "horse"]
  import time
  start = time.perf_counter()
  images = pipe(prompts[:batch]).images
  seconds = time.perf_counter() - start
  print('{:.6f}s for {} images'.format(seconds, len(images)))
  return images

run(pipe, 1)
run(pipe, 1)
run(pipe, 1)
run(pipe, 1)
run(pipe, 1)

images=run(pipe, 4)
images=run(pipe, 4)

images[0].save('tensorrt_mt_fuji.png')
