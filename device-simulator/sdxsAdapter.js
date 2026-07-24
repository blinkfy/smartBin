import { fetchArrayBufferWithCacheProgress, purgeModelCache } from '../cache.js';

const MODEL_ID = 'sdxs-512-dreamshaper';
const MODEL_BASE = '/txt2img-models/sdxs-512-dreamshaper';
const PROMPT_SHAPE = [1, 77, 768];
const PROMPT_LENGTH = 77 * 768;

const SCHEDULER_CONFIG = Object.freeze({
  betaStart: 0.00085,
  betaEnd: 0.012,
  betaSchedule: 'scaled_linear',
  numTrainTimesteps: 1000,
  predictionType: 'epsilon',
  timestepSpacing: 'trailing'
});

const VAE_CONFIG = Object.freeze({
  scalingFactor: 1.0,
  latentMagnitude: 3,
  latentShift: 0.5
});

function buildSingleStepEulerSchedule() {
  const betas = new Float64Array(SCHEDULER_CONFIG.numTrainTimesteps);
  const start = Math.sqrt(SCHEDULER_CONFIG.betaStart);
  const end = Math.sqrt(SCHEDULER_CONFIG.betaEnd);
  let alphaProduct = 1;
  const sigmas = new Float64Array(SCHEDULER_CONFIG.numTrainTimesteps);

  for (let index = 0; index < betas.length; index += 1) {
    const ratio = index / (betas.length - 1);
    const value = start + (end - start) * ratio;
    betas[index] = value * value;
    alphaProduct *= 1 - betas[index];
    sigmas[index] = Math.sqrt((1 - alphaProduct) / alphaProduct);
  }

  const timestep = SCHEDULER_CONFIG.numTrainTimesteps - 1;
  return Object.freeze({ timestep, sigma: sigmas[timestep], sigmaNext: 0 });
}

const SINGLE_STEP_SCHEDULE = buildSingleStepEulerSchedule();

function mulberry32(seed) {
  let value = seed >>> 0;
  return function random() {
    value += 0x6D2B79F5;
    let result = Math.imul(value ^ (value >>> 15), 1 | value);
    result ^= result + Math.imul(result ^ (result >>> 7), 61 | result);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function createNoise(shape, sigma, seed) {
  const random = mulberry32(seed ?? 0);
  const size = shape.reduce((product, value) => product * value, 1);
  const values = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    const first = Math.max(random(), 1e-7);
    const second = random();
    values[index] = Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second) * sigma;
  }
  return values;
}

function scaleEulerInput(values, sigma) {
  const divisor = Math.sqrt(sigma * sigma + 1);
  const scaled = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    scaled[index] = values[index] / divisor;
  }
  return scaled;
}

function eulerSingleStep(sample, noisePrediction, sigma, sigmaNext) {
  const next = new Float32Array(sample.length);
  const delta = sigmaNext - sigma;
  for (let index = 0; index < sample.length; index += 1) {
    next[index] = sample[index] + noisePrediction[index] * delta;
  }
  return next;
}

function normalizeProgress(progress, loaded, total, asset) {
  const resolvedTotal = total || progress.expectedBytes;
  return {
    phase: 'loading',
    message: `downloading ${asset}...`,
    pct: resolvedTotal ? Math.min(100, Math.round((progress.loadedBytes + loaded) / progress.totalBytes * 100)) : undefined,
    bytesDownloaded: progress.loadedBytes + loaded,
    totalBytesExpected: progress.totalBytes,
    asset,
    accuracy: 'approximate'
  };
}

async function fetchPromptEmbedding(prompt, onProgress) {
  onProgress?.({ phase: 'encoding', pct: 15, message: '正在编码自定义文本...' });
  const response = await fetch('/txt2img-embedding', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL_ID, prompt })
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.message || `自定义文本编码失败（${response.status}）`);
  }
  const embedding = new Float32Array(await response.arrayBuffer());
  if (embedding.length !== PROMPT_LENGTH) {
    throw new Error(`自定义文本 embedding 尺寸错误：${embedding.length}`);
  }
  return embedding;
}

async function tensorToPngBlob(tensor) {
  const [, channels, height, width] = tensor.dims;
  if (channels !== 3) throw new Error(`VAE输出通道错误：${channels}`);
  const data = tensor.data;
  const pixels = new Uint8ClampedArray(width * height * 4);
  let outputIndex = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      for (let channel = 0; channel < 3; channel += 1) {
        const value = data[channel * height * width + y * width + x];
        pixels[outputIndex++] = Math.round(Math.max(0, Math.min(1, value / 2 + 0.5)) * 255);
      }
      pixels[outputIndex++] = 255;
    }
  }
  const imageData = new ImageData(pixels, width, height);
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas 2D context unavailable');
  context.putImageData(imageData, 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

export class SDXSAdapter {
  constructor() {
    this.id = MODEL_ID;
    this.loaded = false;
    this.ort = null;
    this.sessions = {};
    this.modelBase = MODEL_BASE;
  }

  checkSupport(capabilities) {
    return capabilities.webgpu && capabilities.shaderF16 ? ['webgpu'] : [];
  }

  async load(options) {
    if (!options.backendPreference.includes('webgpu')) {
      return { ok: false, reason: 'backend_unavailable', message: 'SDXS FP16需要WebGPU' };
    }
    if (options.modelBaseUrl) this.modelBase = options.modelBaseUrl;

    try {
      const ortModule = await import('onnxruntime-web/webgpu');
      this.ort = ortModule.default ?? ortModule;
      this.ort.env.logLevel = 'warning';
    } catch (error) {
      return { ok: false, reason: 'internal_error', message: `onnxruntime-web加载失败：${error.message || error}` };
    }

    const models = {
      unet: { url: 'unet/model.fp16.onnx', sizeMB: 658 },
      vae_decoder: { url: 'vae_decoder/model.fp16.onnx', sizeMB: 3 }
    };
    const totalBytes = Object.values(models).reduce((sum, model) => sum + model.sizeMB * 1024 * 1024, 0);
    const progress = { loadedBytes: 0, totalBytes, expectedBytes: totalBytes };
    const sessionOptions = {
      executionProviders: ['webgpu'],
      enableMemPattern: false,
      enableCpuMemArena: false,
      graphOptimizationLevel: 'all'
    };

    try {
      options.onProgress?.({ phase: 'loading', message: '正在加载SDXS单步模型...', pct: 0, bytesDownloaded: 0, totalBytesExpected: totalBytes });
      for (const [key, model] of Object.entries(models)) {
        const expectedBytes = model.sizeMB * 1024 * 1024;
        const buffer = await fetchArrayBufferWithCacheProgress(
          `${this.modelBase}/${model.url}`,
          this.id,
          (loaded, total) => options.onProgress?.(normalizeProgress(progress, loaded, total || expectedBytes, model.url)),
          expectedBytes
        );
        progress.loadedBytes += buffer.byteLength;
        this.sessions[key] = await this.ort.InferenceSession.create(buffer, sessionOptions);
      }
      this.loaded = true;
      return { ok: true, backendUsed: 'webgpu', bytesDownloaded: progress.loadedBytes };
    } catch (error) {
      await this.unload();
      return { ok: false, reason: 'internal_error', message: `SDXS模型加载失败：${error.message || error}` };
    }
  }

  isLoaded() {
    return this.loaded;
  }

  async generate(params) {
    if (!this.loaded) return { ok: false, reason: 'model_not_loaded', message: '请先加载SDXS模型' };
    const { prompt, promptEmbedding, width = 512, height = 512, signal, onProgress, seed = 0 } = params;
    if ((!prompt || !prompt.trim()) && !promptEmbedding) {
      return { ok: false, reason: 'unsupported_option', message: 'Prompt is required' };
    }
    if (width !== 512 || height !== 512) {
      return { ok: false, reason: 'unsupported_option', message: 'SDXS当前仅支持512×512' };
    }

    const startedAt = performance.now();
    const ort = this.ort;
    let embeddingTensor;
    let latentTensor;
    let modelInputTensor;
    let timestepTensor;
    let denoisedTensor;
    try {
      const embedding = promptEmbedding
        ? (promptEmbedding instanceof Float32Array ? promptEmbedding : new Float32Array(promptEmbedding))
        : await fetchPromptEmbedding(prompt.trim(), onProgress);
      if (embedding.length !== PROMPT_LENGTH) {
        throw new Error(`Prompt embedding必须为[1,77,768]，实际长度${embedding.length}`);
      }
      if (signal?.aborted) return { ok: false, reason: 'cancelled' };

      embeddingTensor = new ort.Tensor('float32', embedding, PROMPT_SHAPE);
      const latentShape = [1, 4, 64, 64];
      const schedule = SINGLE_STEP_SCHEDULE;
      const latentValues = createNoise(latentShape, schedule.sigma, seed);
      latentTensor = new ort.Tensor('float32', latentValues, latentShape);
      modelInputTensor = new ort.Tensor('float32', scaleEulerInput(latentValues, schedule.sigma), latentShape);
      timestepTensor = new ort.Tensor('float32', Float32Array.of(schedule.timestep), [1]);

      onProgress?.({ phase: 'denoising', pct: 55, message: 'SDXS单步去噪...' });
      const unetResult = await this.sessions.unet.run({
        sample: modelInputTensor,
        timestep: timestepTensor,
        encoder_hidden_states: embeddingTensor
      });
      const noisePrediction = unetResult.out_sample ?? unetResult.sample ?? Object.values(unetResult)[0];
      const denoised = eulerSingleStep(latentValues, noisePrediction.data, schedule.sigma, schedule.sigmaNext);
      const vaeInput = new Float32Array(denoised.length);
      for (let index = 0; index < denoised.length; index += 1) {
        vaeInput[index] = denoised[index] / VAE_CONFIG.scalingFactor;
      }
      denoisedTensor = new ort.Tensor('float32', vaeInput, latentShape);

      onProgress?.({ phase: 'decoding', pct: 90, message: '轻量VAE解码...' });
      const vaeResult = await this.sessions.vae_decoder.run({ latent_sample: denoisedTensor });
      const imageTensor = vaeResult.sample ?? Object.values(vaeResult)[0];
      const blob = await tensorToPngBlob(imageTensor);
      imageTensor.dispose?.();
      noisePrediction.dispose?.();
      const timeMs = performance.now() - startedAt;
      onProgress?.({ phase: 'complete', pct: 100, timeMs });
      return { ok: true, blob, timeMs };
    } catch (error) {
      return { ok: false, reason: 'internal_error', message: error.message || String(error) };
    } finally {
      embeddingTensor?.dispose?.();
      latentTensor?.dispose?.();
      modelInputTensor?.dispose?.();
      timestepTensor?.dispose?.();
      denoisedTensor?.dispose?.();
    }
  }

  async unload() {
    this.sessions.unet?.release?.();
    this.sessions.vae_decoder?.release?.();
    this.sessions = {};
    this.ort = null;
    this.loaded = false;
  }

  async purgeCache() {
    await purgeModelCache(this.id);
  }
}
