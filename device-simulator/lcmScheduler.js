// LCMScheduler (epsilon, scaled_linear), matching Diffusers' discrete boundary conditions.
export function createLcmSchedule(config, steps) {
  const trainSteps = config.num_train_timesteps;
  const originalSteps = config.original_inference_steps;
  if (!Number.isInteger(steps) || steps < 1 || steps > originalSteps) {
    throw new Error(`LCM步数必须为1～${originalSteps}的整数`);
  }
  if (config.beta_schedule !== 'scaled_linear' || config.prediction_type !== 'epsilon') {
    throw new Error('不支持的LCM调度配置');
  }
  const alphas = new Float32Array(trainSteps);
  const start = Math.sqrt(config.beta_start);
  const end = Math.sqrt(config.beta_end);
  let product = 1;
  for (let i = 0; i < trainSteps; i += 1) {
    const root = Math.fround(start + (end - start) * i / (trainSteps - 1));
    const alpha = Math.fround(1 - Math.fround(root * root));
    product *= alpha;
    alphas[i] = product;
  }
  const ratio = Math.floor(trainSteps / originalSteps);
  const timesteps = Array.from({ length: steps }, (_, i) =>
    (originalSteps - Math.floor(i * originalSteps / steps)) * ratio - 1);
  return { timesteps, alphas, timestepScaling: config.timestep_scaling };
}

export function lcmStep(sample, prediction, schedule, index, noise) {
  const timestep = schedule.timesteps[index];
  const alpha = schedule.alphas[timestep];
  const sqrtAlpha = Math.sqrt(alpha);
  const sqrtBeta = Math.sqrt(1 - alpha);
  const scaled = timestep * schedule.timestepScaling;
  const cSkip = 0.25 / (scaled * scaled + 0.25);
  const cOut = scaled / Math.sqrt(scaled * scaled + 0.25);
  const last = index === schedule.timesteps.length - 1;
  if (!last && (!noise || noise.length !== sample.length)) throw new Error('LCM噪声尺寸错误');
  const nextAlpha = last ? 1 : schedule.alphas[schedule.timesteps[index + 1]];
  const result = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i += 1) {
    const original = (sample[i] - sqrtBeta * prediction[i]) / sqrtAlpha;
    const denoised = cOut * original + cSkip * sample[i];
    result[i] = last ? denoised : Math.sqrt(nextAlpha) * denoised + Math.sqrt(1 - nextAlpha) * noise[i];
  }
  return result;
}
