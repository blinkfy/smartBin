#!/usr/bin/env python3
"""Compare browser LCM math with Diffusers for 1/2/4/8 steps, including intermediate noise."""

import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch
from diffusers import LCMScheduler


ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "models/lcm-tiny-sd/scheduler_config.json").read_text(encoding="utf-8"))
    cases = []
    expected = []
    rng = np.random.default_rng(42)
    for steps in (1, 2, 4, 8):
        scheduler = LCMScheduler(**config)
        scheduler.set_timesteps(steps)
        initial = rng.normal(size=32).astype(np.float32)
        sample = torch.from_numpy(initial.copy())
        predictions = rng.normal(size=(steps, 32)).astype(np.float32)
        noises = rng.normal(size=(steps, 32)).astype(np.float32)
        for i, timestep in enumerate(scheduler.timesteps):
            with patch("diffusers.schedulers.scheduling_lcm.randn_tensor", return_value=torch.from_numpy(noises[i])):
                sample = scheduler.step(torch.from_numpy(predictions[i]), timestep, sample).prev_sample
        cases.append({"steps": steps, "initial": initial.tolist(), "predictions": predictions.tolist(), "noises": noises.tolist()})
        expected.append((scheduler.timesteps.tolist(), sample.numpy()))
    code = """
import { createLcmSchedule, lcmStep } from './lcmScheduler.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const { config, cases } = JSON.parse(input);
const results = cases.map(test => {
  const schedule = createLcmSchedule(config, test.steps);
  let sample = new Float32Array(test.initial);
  for (let i = 0; i < test.steps; i++) sample = lcmStep(sample, test.predictions[i], schedule, i, test.noises[i]);
  return { timesteps: schedule.timesteps, sample: Array.from(sample) };
});
for (const steps of [0, 51, 2.5]) {
  let rejected = false; try { createLcmSchedule(config, steps); } catch { rejected = true; }
  if (!rejected) throw new Error('Invalid steps accepted');
}
console.log(JSON.stringify(results));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", code], cwd=ROOT,
                            input=json.dumps({"config": config, "cases": cases}), text=True,
                            capture_output=True, check=True)
    for case, actual, (timesteps, sample) in zip(cases, json.loads(result.stdout), expected):
        assert actual["timesteps"] == timesteps
        np.testing.assert_allclose(actual["sample"], sample, rtol=2e-5, atol=2e-5)
        delta = np.max(np.abs(np.array(actual["sample"]) - sample))
        print(f"LCM_{case['steps']}_STEPS_OK max_abs={delta:.8f}")


if __name__ == "__main__":
    main()
