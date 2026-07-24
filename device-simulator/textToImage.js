import { Txt2ImgWorkerClient } from '/txt2img-vendor/web-txt2img/dist/index.js';

const MODEL_ID = 'sdxs-512-dreamshaper';
const WEBGPU_UNSUPPORTED_MESSAGE = '当前浏览器或设备不支持WebGPU，请使用最新版Chrome或Edge并开启硬件加速';
const EMBEDDING_LENGTH = 77 * 768;

export const TEXT_TO_IMAGE_PRESETS = [
  {
    id: 'plastic-bottle',
    label: '塑料瓶',
    category: 'recyclable',
    item: '塑料瓶',
    prompt: 'single empty plastic bottle without any others, pure white background',
    embeddingUrl: '/txt2img-presets/plastic-bottle.f32?v=sdxs-simple-white-v3'
  },
  {
    id: 'medicine-blister',
    label: '废药板',
    category: 'hazardous',
    item: '废药板',
    prompt: 'pill transparent tablet with pills, without any others, pure white background',
    embeddingUrl: '/txt2img-presets/medicine-blister.f32?v=sdxs-simple-white-v3'
  },
  {
    id: 'vegetable',
    label: '蔬菜',
    category: 'kitchen',
    item: '菜叶',
    prompt: 'a vegetable, without any other objects, pure white background',
    embeddingUrl: '/txt2img-presets/vegetable.f32?v=sdxs-simple-white-v3'
  }
];

let client = null;
let capabilitiesPromise = null;
let loadPromise = null;
let modelLoaded = false;
const embeddingPromises = new Map();

function getClient() {
  if (!client) {
    client = Txt2ImgWorkerClient.createDefault();
  }
  return client;
}

function normalizeProgress(progress) {
  if (!progress || typeof progress !== 'object') return {};
  return {
    phase: progress.phase || '',
    message: progress.message || '',
    pct: typeof progress.pct === 'number' ? progress.pct : null,
    bytesDownloaded: progress.bytesDownloaded,
    totalBytesExpected: progress.totalBytesExpected
  };
}

function reportProgress(callback, fallbackMessage, progress) {
  if (typeof callback !== 'function') return;
  const normalized = normalizeProgress(progress);
  callback({
    ...normalized,
    message: normalized.message || fallbackMessage
  });
}

function isWebGpuDeviceError(error) {
  const message = String(error?.message || error || '');
  return /WebGPU validation failed|Invalid Buffer|device lost|GPUDevice/i.test(message);
}

function resetAfterWebGpuError(error) {
  if (!isWebGpuDeviceError(error)) return false;
  disposeTextToImage();
  return true;
}

async function ensureWebGpu() {
  if (!navigator.gpu) {
    throw new Error(WEBGPU_UNSUPPORTED_MESSAGE);
  }

  if (!capabilitiesPromise) {
    capabilitiesPromise = getClient().detect();
  }

  const caps = await capabilitiesPromise;
  if (!caps || !caps.webgpu) {
    throw new Error(WEBGPU_UNSUPPORTED_MESSAGE);
  }
  return caps;
}

export async function loadTextToImageModel(onProgress) {
  await ensureWebGpu();
  if (modelLoaded) return { ok: true };

  if (!loadPromise) {
    loadPromise = getClient()
      .load(
        MODEL_ID,
        { backendPreference: ['webgpu'] },
        progress => reportProgress(onProgress, '正在下载SDXS单步模型...', progress)
      )
      .then(result => {
        if (!result || result.ok === false) {
          throw new Error(result?.message || 'SDXS模型加载失败');
        }
        modelLoaded = true;
        reportProgress(onProgress, '模型已就绪', { phase: 'ready', pct: 100 });
        return result;
      })
      .catch(error => {
        loadPromise = null;
        throw error;
      });
  }

  return loadPromise;
}

async function loadPresetEmbedding(presetId, onProgress) {
  const preset = TEXT_TO_IMAGE_PRESETS.find(item => item.id === presetId);
  if (!preset) throw new Error('未找到所选预置示例');

  if (!embeddingPromises.has(preset.id)) {
    embeddingPromises.set(preset.id, fetch(preset.embeddingUrl)
      .then(response => {
        if (!response.ok) throw new Error(`预置词嵌入加载失败（${response.status}）`);
        return response.arrayBuffer();
      })
      .then(buffer => {
        const embedding = new Float32Array(buffer);
        if (embedding.length !== EMBEDDING_LENGTH) {
          throw new Error(`预置词嵌入尺寸错误：${embedding.length}`);
        }
        return embedding;
      })
      .catch(error => {
        embeddingPromises.delete(preset.id);
        throw error;
      }));
  }

  reportProgress(onProgress, '正在读取预置词嵌入...', { phase: 'encoding', pct: 20 });
  return embeddingPromises.get(preset.id);
}

export function preloadTextToImageModel(onProgress) {
  return loadTextToImageModel(onProgress);
}

export async function textToImage(prompt, onProgress, options = {}) {
  const cleanPrompt = String(prompt || '').trim();
  if (!cleanPrompt) return null;

  await loadTextToImageModel(onProgress);
  const promptEmbedding = options.presetId
    ? await loadPresetEmbedding(options.presetId, onProgress)
    : null;

  const task = getClient().generate(
    {
      prompt: cleanPrompt,
      ...(promptEmbedding ? { promptEmbedding } : {}),
      width: 512,
      height: 512,
      seed: Number.isInteger(options.seed)
        ? options.seed
        : Math.floor(Math.random() * 2147483647)
    },
    progress => reportProgress(onProgress, '正在生成图片...', progress)
  );

  let result;
  try {
    result = await task.promise;
  } catch (error) {
    if (resetAfterWebGpuError(error)) {
      throw new Error('WebGPU资源异常，推理引擎已自动重置，请再次点击生成');
    }
    throw error;
  }
  if (!result || result.ok === false) {
    const error = new Error(result?.message || '图片生成失败');
    if (resetAfterWebGpuError(error)) {
      throw new Error('WebGPU资源异常，推理引擎已自动重置，请再次点击生成');
    }
    throw error;
  }
  if (!result.blob) {
    throw new Error('图片生成失败：未收到图片数据');
  }
  return result.blob;
}

export function disposeTextToImage() {
  if (client) {
    client.terminate();
    client = null;
  }
  capabilitiesPromise = null;
  loadPromise = null;
  modelLoaded = false;
  embeddingPromises.clear();
}
