# Apollo Optimizer in Tensorflow 2.x

### Notes:

- Warmup is important with Apollo optimizer, so be sure to pass in a learning rate schedule vs. a constant learning 
rate for `learning_rate`. One cycle scheduler is given as an example in one_cycle_lr_schedule.py
- To clip gradient norms as in paper, add either `clipnorm` (parameter-wise clipping by norm) or `global_clipnorm` to 
the arguments (for example `clipnorm=0.1`).
- Decoupled weight decay is used by default.
