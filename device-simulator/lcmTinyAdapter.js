import { SDXSAdapter, createNoise, mulberry32, fetchPromptEmbedding, tensorToPngBlob } from './sd-turbo.js';
import { createLcmSchedule, lcmStep } from '/lcmScheduler.js';

export class LCMTinyAdapter extends SDXSAdapter {
  constructor() {
    super();
    this.id = 'lcm-tiny-sd';
    this.displayName = 'LCM-Tiny-SD';
    this.modelBase = '/txt2img-models/lcm-tiny-sd';
    this.modelAssets = {
      unet: { url: 'unet/model.fp16.onnx', sizeMB: 0.34,
        externalData: { url: 'unet/model.onnx_data', path: 'model.onnx_data', sizeMB: 617 } },
      vae_decoder: { url: '/txt2img-models/sdxs-512-dreamshaper/vae_decoder/model.fp16.onnx', sizeMB: 2.35 }
    };
    this.scheduleConfig = null;
  }

  async load(options) {
    if (options.modelBaseUrl) this.modelBase = options.modelBaseUrl;
    try {
      const response = await fetch(`${this.modelBase}/scheduler_config.json`, { cache: 'no-cache' });
      if (!response.ok) throw new Error('缺少LCM模型，请先运行 scripts/prepare_lcm_tiny_sd.py');
      this.scheduleConfig = await response.json();
      createLcmSchedule(this.scheduleConfig, 4);
      return await super.load(options);
    } catch (error) {
      return { ok: false, reason: 'internal_error', message: error.message || String(error) };
    }
  }

  async generate(params) {
    if (!this.loaded) return { ok: false, reason: 'model_not_loaded', message: '请先加载LCM-Tiny-SD模型' };
    const { prompt, promptEmbedding, width = 512, height = 512, seed = 0,
      steps = 4, signal, onProgress } = params;
    if (width !== 512 || height !== 512) return { ok: false, reason: 'unsupported_option', message: 'LCM当前仅支持512×512' };
    const startedAt = performance.now();
    const ort = this.ort;
    let embeddingTensor;
    let inputTensor;
    let timestepTensor;
    let predictionTensor;
    let decodedTensor;
    try {
      const schedule = createLcmSchedule(this.scheduleConfig, steps);
      const embedding = promptEmbedding
        ? new Float32Array(promptEmbedding)
        : await fetchPromptEmbedding(String(prompt || '').trim(), onProgress, this.id);
      if (embedding.length !== 77 * 768) throw new Error('LCM文本嵌入尺寸错误');
      embeddingTensor = new ort.Tensor('float32', embedding, [1, 77, 768]);
      const shape = [1, 4, 64, 64];
      const random = mulberry32(seed);
      let latent = createNoise(shape, 1, seed, random);
      for (let index = 0; index < steps; index += 1) {
        if (signal?.aborted) return { ok: false, reason: 'cancelled' };
        onProgress?.({ phase: 'denoising', pct: 25 + Math.round(index / steps * 60),
          message: `LCM-Tiny-SD去噪 ${index + 1}/${steps} 步` });
        inputTensor = new ort.Tensor('float32', latent, shape);
        timestepTensor = new ort.Tensor('float32', Float32Array.of(schedule.timesteps[index]), [1]);
        const result = await this.sessions.unet.run({ sample: inputTensor,
          timestep: timestepTensor, encoder_hidden_states: embeddingTensor });
        predictionTensor = result.out_sample ?? Object.values(result)[0];
        const noise = index < steps - 1 ? createNoise(shape, 1, seed, random) : null;
        latent = lcmStep(latent, predictionTensor.data, schedule, index, noise);
        predictionTensor.dispose?.(); predictionTensor = null;
        inputTensor.dispose?.(); inputTensor = null;
        timestepTensor.dispose?.(); timestepTensor = null;
      }
      if (signal?.aborted) return { ok: false, reason: 'cancelled' };
      // TAESD consumes the diffusion latents directly (scaling_factor=1.0).
      inputTensor = new ort.Tensor('float32', latent, shape);
      onProgress?.({ phase: 'decoding', pct: 90, message: '轻量VAE解码...' });
      const result = await this.sessions.vae_decoder.run({ latent_sample: inputTensor });
      decodedTensor = result.sample ?? Object.values(result)[0];
      const blob = await tensorToPngBlob(decodedTensor);
      const timeMs = performance.now() - startedAt;
      onProgress?.({ phase: 'complete', pct: 100, timeMs });
      return { ok: true, blob, timeMs };
    } catch (error) {
      return { ok: false, reason: 'internal_error', message: error.message || String(error) };
    } finally {
      for (const tensor of [embeddingTensor, inputTensor, timestepTensor, predictionTensor, decodedTensor]) {
        tensor?.dispose?.();
      }
    }
  }
}
