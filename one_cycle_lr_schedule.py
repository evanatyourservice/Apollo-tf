import tensorflow as tf
import math


class OneCycleScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, lr_max, steps, phase_1_pct=0.05, div_factor=25.0):
    super(OneCycleScheduler, self).__init__()

    self.lr_max = lr_max
    self.steps = steps
    self.phase_1_pct = phase_1_pct
    self.div_factor = div_factor

    self.lr_min = lr_max / div_factor
    self.final_lr = lr_max / (div_factor * 1.0e4)
    self.phase_1_steps = round(steps * phase_1_pct) - 1
    self.phase_2_steps = round(steps - self.phase_1_steps) - 2

  @tf.function
  def __call__(self, global_step):
    if global_step > self.phase_1_steps:
      return self.calc_lr(global_step - self.phase_1_steps, self.lr_max, self.final_lr, self.phase_2_steps)
    else:
      return self.calc_lr(global_step, self.lr_min, self.lr_max, self.phase_1_steps)

  @tf.function
  def calc_lr(self, n, start, end, steps):
    cos = tf.math.cos(math.pi * (n / steps)) + 1.0
    return end + (start - end) / 2.0 * cos

  def get_config(self):
    config = {
        "lr_max": self.lr_max,
        "steps": self.steps,
        "phase_1_pct": self.phase_1_pct,
        "div_factor": self.div_factor,
    }
    return config