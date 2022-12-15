"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, p in enumerate(self.params):
            if p.grad is not None:

                # new Tensor for unifying the dtype
                # grad.dtype could be float64 and p.dtype could be float32
                # d_p = p.grad.detach()
                d_p = ndl.Tensor(p.grad.detach(), dtype=p.dtype)  # Gradient

                if self.weight_decay != 0:
                    d_p += self.weight_decay * p.detach()

                if self.momentum != 0:
                    u_prev = self.u.get(i, ndl.init.zeros(
                        *d_p.shape, device=d_p.device, dtype=d_p.dtype))
                    self.u[i] = self.momentum * \
                        u_prev + (1 - self.momentum) * d_p
                    d_p = self.u[i].detach()

                p.data = p.data - self.lr * d_p.detach()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array(
            [np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # new Tensor for unifying the dtype
                # grad.dtype could be float64 and p.dtype could be float32
                # d_p = p.grad.detach()
                d_p = ndl.Tensor(p.grad.detach(), dtype=p.dtype)  # Gradient

                if self.weight_decay != 0:
                    d_p += self.weight_decay * p.detach()

                u_prev = self.m.get(i, ndl.init.zeros(
                    *d_p.shape, device=d_p.device, dtype=d_p.dtype))
                v_prev = self.v.get(i, ndl.init.zeros(
                    *d_p.shape, device=d_p.device, dtype=d_p.dtype))

                u_cur = self.beta1 * u_prev + (1 - self.beta1) * d_p
                v_cur = self.beta2 * v_prev + (1 - self.beta2) * (d_p ** 2)

                self.m[i] = u_cur.detach()
                self.v[i] = v_cur.detach()

                u_prev_correct = u_cur.detach() / (1 - self.beta1 ** self.t)
                v_prev_correct = v_cur.detach() / (1 - self.beta2 ** self.t)

                p.data = p.data - self.lr * u_prev_correct.detach() / (v_prev_correct **
                                                                       (1/2) + self.eps)
        ### END YOUR SOLUTION
