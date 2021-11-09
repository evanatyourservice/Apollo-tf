import tensorflow as tf
import numpy as np


class Apollo(tf.keras.optimizers.Optimizer):
    """
    Implements Apollo algorithm.
        Arguments:
            learning_rate (float): learning rate (default: 0.01)
            beta (float, optional): coefficient used for computing running averages of gradient (default: 0.9)
            epsilon (float, optional): term added to the denominator to improve numerical stability (default: 1e-4)
            rebound (str, optional): rectified bound for diagonal hessian, 'belief' for scale-invariant:
                ``'constant'`` | ``'belief'`` (default: 'constant')
            weight_decay (float, optional): weight decay coefficient (default: 0.0)
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
            self,
            learning_rate=0.01,
            beta=0.9,
            epsilon=1e-4,
            rebound='constant',
            weight_decay=0.0,
            name="ApolloOptimizer",
            **kwargs):
        super().__init__(name, **kwargs)
        assert rebound in ['constant', 'belief'], "rebound = 'constant' for normal or 'belief' for scale-invariant"
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta", beta)
        self.epsilon = epsilon
        self.rebound_type = rebound
        self._set_hyper("weight_decay", weight_decay)
        self._has_weight_decay = weight_decay != 0.0

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "exp_avg_grad")
        for var in var_list:
            self.add_slot(var, "approx_hessian")
        for var in var_list:
            self.add_slot(var, "update")

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        curr_lr = self._decayed_lr(var_dtype)
        weight_decay = self._decayed_wd(var_dtype)
        beta = self._get_hyper("beta", var_dtype)
        eps = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)

        exp_avg_grad = self.get_slot(var, "exp_avg_grad")
        B = self.get_slot(var, "approx_hessian")
        d_p = self.get_slot(var, "update")

        bias_correction = 1 - beta ** local_step
        alpha = (1 - beta) / bias_correction

        # calc the diff grad
        delta_grad = grad - exp_avg_grad
        if self.rebound_type == 'belief':
            rebound = tf.norm(delta_grad, ord=np.inf)
        else:
            rebound = 0.01
            eps = eps / rebound

        # Update the running average grad
        exp_avg_grad_t = exp_avg_grad.assign_add(delta_grad * alpha, use_locking=self._use_locking)

        denom = tf.math.pow(
            tf.math.reduce_sum(tf.math.pow(tf.math.abs(d_p), tf.constant(4.0))), tf.constant(1.0/4.0)) + eps
        d_p_t = d_p / denom
        v_sq = d_p_t ** 2
        delta = tf.math.reduce_sum((delta_grad / denom) * d_p_t) * -alpha - tf.math.reduce_sum(B * v_sq)

        # Update B
        B_t = B.assign_add(v_sq * delta, use_locking=self._use_locking)

        # calc direction of parameter updates
        if self.rebound_type == 'belief':
            denom = tf.math.maximum(tf.math.abs(B_t), rebound) + eps / alpha
        else:
            denom = tf.math.maximum(tf.math.abs(B_t), rebound)

        d_p_t = exp_avg_grad_t / denom

        # Perform step weight decay (decoupled)
        if self._has_weight_decay:
            d_p_t = d_p.assign(d_p_t + var * weight_decay)
        else:
            d_p_t = d_p.assign(d_p_t)

        var.assign_add(d_p_t * -curr_lr, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices):
        raise "This implementation of Apollo does not support sparse gradients"

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta": self._serialize_hyperparameter("beta"),
                "epsilon": self.epsilon,
                "rebound": self.rebound_type,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
            }
        )
        return config
